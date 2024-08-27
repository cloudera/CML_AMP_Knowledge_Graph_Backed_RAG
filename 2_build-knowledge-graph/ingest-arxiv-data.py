import os

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.graphs import Neo4jGraph
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores.neo4j_vector import Neo4jVector

import utils.constants as const
from utils.arxiv_utils import create_paper_object_from_arxiv_id
from utils.data_utils import (
    create_cypher_batch_query_to_create_citation_relationship,
    create_cypher_batch_query_to_insert_arxiv_papers,
    create_indices_queries,
    create_query_for_category_insertion,
)
from utils.huggingface_utils import cache_and_load_embedding_model
from utils.neo4j_utils import (
    get_neo4j_credentails,
    is_neo4j_server_up,
    reset_neo4j_server,
    wait_for_neo4j_server,
)

load_dotenv()

if not is_neo4j_server_up():
    reset_neo4j_server()
    wait_for_neo4j_server()

graph = Neo4jGraph(
    username=get_neo4j_credentails()["username"],
    password=get_neo4j_credentails()["password"],
    url=get_neo4j_credentails()["uri"],
)
graph.query("MATCH (n) DETACH DELETE n")
graph.query(create_query_for_category_insertion())

embedding = cache_and_load_embedding_model()

Neo4jVector.from_existing_graph(
    embedding=embedding,
    url=get_neo4j_credentails()["uri"],
    username=get_neo4j_credentails()["username"],
    password=get_neo4j_credentails()["password"],
    index_name="category_embedding_index",
    node_label="Category",
    text_node_properties=["title", "description"],
    embedding_node_property="embedding",
)

for q in create_indices_queries():
    graph.query(q)

arxiv_ids_set = set(const.seed_arxiv_paper_ids)
arxiv_ids_set.update(
    [
        cited_paper
        for cited_papers in [
            create_paper_object_from_arxiv_id(seed_paper_id).cited_arxiv_papers
            for seed_paper_id in const.seed_arxiv_paper_ids
        ]
        for cited_paper in cited_papers
    ]
)
print(f"Total arxiv papers to insert: {len(arxiv_ids_set)}")
papers_to_insert = list()
for arxiv_id in arxiv_ids_set:
    try:
        papers_to_insert.append(create_paper_object_from_arxiv_id(arxiv_id))
        print(f"arxiv paper {arxiv_id} added to the list.")
    except Exception as e:
        print(f"Error in creating paper object for arxiv_id {arxiv_id}: {e}")

paper_batch = list()
batch_size = 10
for i, paper in enumerate(papers_to_insert):
    if len(paper_batch) < batch_size and i != len(papers_to_insert) - 1:
        paper_batch.append(paper)
        continue
    query = create_cypher_batch_query_to_insert_arxiv_papers(paper_batch)
    try:
        graph.query(query)
        print(f"Inserted papers {[p.arxiv_id for p in paper_batch]}")
    except Exception as e:
        for p in paper_batch:
            query = create_cypher_batch_query_to_insert_arxiv_papers([p])
            try:
                graph.query(query)
                print(f"Inserted paper {p.arxiv_id}")
            except Exception as e:
                print(f"Error in inserting paper {p.arxiv_id}")
    paper_batch.clear()


# create citation relationships
for paper in papers_to_insert:
    query = create_cypher_batch_query_to_create_citation_relationship(paper.arxiv_id)
    graph.query(query)
    print(f"Created citation relationships for paper {paper.arxiv_id}")

raw_docs = [
    Document(page_content=p.full_text, metadata={"arxiv_id": p.arxiv_id})
    for p in papers_to_insert
]
# Define chunking strategy
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=20,
    disallowed_special=(),
)
# Chunk the document
documents = text_splitter.split_documents(raw_docs)
print(f"Number of chunks to be inserted into the knowledge graph: {len(documents)}")

document_batch = list()
batch_size = 50
for i, doc in enumerate(documents):
    document_batch.append(doc)
    if len(document_batch) < batch_size and i != len(documents) - 1:
        continue
    try:
        Neo4jVector.from_documents(
            documents=document_batch,
            embedding=embedding,
            url=get_neo4j_credentails()["uri"],
            username=get_neo4j_credentails()["username"],
            password=get_neo4j_credentails()["password"],
        )
        print(f"Inserted chunk {i+1}/{len(documents)}")
    except Exception as e:
        for d in document_batch:
            try:
                Neo4jVector.from_documents(
                    documents=[d],
                    embedding=embedding,
                    url=get_neo4j_credentails()["uri"],
                    username=get_neo4j_credentails()["username"],
                    password=get_neo4j_credentails()["password"],
                )
            except Exception as e:
                print(f"Error in inserting chunk: {e}")
    document_batch.clear()

# Link the chunks to the papers
graph.query(
    """
MATCH (p:Paper), (c:Chunk)
WHERE p.id = c.arxiv_id
MERGE (p)-[:CONTAINS_TEXT]->(c)
"""
)

# Delete orphan chunks with no papers
graph.query(
    """
MATCH (c:Chunk)
WHERE NOT (c)<-[:CONTAINS_TEXT]-()
DETACH DELETE c
"""
)

# Get the number of chunks finally present in the DB
chunk_count = graph.query(
    """
MATCH (c:Chunk)
RETURN COUNT(c) as chunk_count
"""
)[0]["chunk_count"]

print(f"Number of chunks in the inserted into the knowledge graph: {chunk_count}")
