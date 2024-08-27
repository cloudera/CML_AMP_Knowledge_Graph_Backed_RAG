from typing import List

from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from ragatouille import RAGPretrainedModel

import utils.constants as const
from utils.arxiv_utils import IngestablePaper, PaperChunk


def get_papers(
    arxiv_ids: List[str] | str, graphDbInstance: Neo4jGraph
) -> List[IngestablePaper]:
    arxiv_ids = arxiv_ids if isinstance(arxiv_ids, list) else [arxiv_ids]
    query = r"""MATCH (p:Paper)
    WHERE p.id IN $list
    RETURN {
    id: p.id, title: p.title, summary: p.summary, published: p.published, arxiv_link: p.arxiv_link, pdf_link: p.pdf_link, cited_arxiv_papers: p.cited_arxiv_papers,
    authors: COLLECT { MATCH (p)-->(a:Author) RETURN a.name },
    categories: COLLECT { MATCH (p)-->(c:Category) RETURN c.code },
    citations: COUNT { (p)<-[:CITES]-(:Paper) }
    } AS result
    """.replace(
        "$list", str(arxiv_ids)
    )
    results = graphDbInstance.query(query)
    papers = list()
    for r in results:
        obj = r["result"]
        paper = IngestablePaper(
            arxiv_id=obj["id"],
            title=obj["title"],
            summary=obj["summary"],
            published_date=obj["published"].to_native(),
            arxiv_link=obj["arxiv_link"],
            pdf_link=obj["pdf_link"],
            authors=obj["authors"],
            categories=obj["categories"],
            cited_arxiv_papers=obj["cited_arxiv_papers"],
            full_text="",
        )
        paper.citation_count = obj["citations"]
        papers.append(paper)
    return papers


def vanilla_retreiver(
    query: str, top_k: int, graphDbInstance: Neo4jGraph, document_index: Neo4jVector
) -> List[PaperChunk]:
    retrieved_chunks = document_index.similarity_search(query=query, k=top_k)
    return [
        PaperChunk(
            text=c.page_content,
            paper=get_papers(
                arxiv_ids=c.metadata["arxiv_id"], graphDbInstance=graphDbInstance
            )[0],
        )
        for c in retrieved_chunks
    ]


def colbert_based_retreiver(
    query: str, top_k: int, graphDbInstance: Neo4jGraph, document_index: Neo4jVector
) -> List[PaperChunk]:
    RAG = RAGPretrainedModel.from_pretrained(const.colbert_model)
    retrieved_chunks = document_index.similarity_search(query=query, k=2 * top_k)
    reranked_results = RAG.rerank(
        query=query, documents=[c.page_content for c in retrieved_chunks], k=top_k
    )
    results = list()
    for r in reranked_results:
        retrieved_chunk = [
            c for c in retrieved_chunks if c.page_content == r["content"]
        ][0]
        chunk = PaperChunk(
            text=r["content"],
            paper=get_papers(
                arxiv_ids=retrieved_chunk.metadata["arxiv_id"],
                graphDbInstance=graphDbInstance,
            )[0],
        )
        chunk.metadata = {"colbert_score": r["score"], "colbert_rank": r["rank"]}
        results.append(chunk)
    return results


def hybrid_retreiver(
    query: str, top_k: int, graphDbInstance: Neo4jGraph, document_index: Neo4jVector
) -> List[PaperChunk]:
    colbert_score_weight, citation_count_weight = 0.5, 0.5
    colbert_results = colbert_based_retreiver(
        query=query,
        top_k=2 * top_k,
        graphDbInstance=graphDbInstance,
        document_index=document_index,
    )
    max_colbert_score = max([c.metadata["colbert_score"] for c in colbert_results])
    max_citation_count = max([c.paper.citation_count for c in colbert_results])

    for c in colbert_results:
        c.metadata.update(
            {
                "hybrid_score": colbert_score_weight
                * c.metadata["colbert_score"]
                / max_colbert_score
                + citation_count_weight * c.paper.citation_count / max_citation_count
            }
        )

    return sorted(
        colbert_results, key=lambda x: x.metadata["hybrid_score"], reverse=True
    )[:top_k]
