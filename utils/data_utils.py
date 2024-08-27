from typing import List

import requests
from bs4 import BeautifulSoup

from utils.arxiv_utils import IngestablePaper


def create_query_for_category_insertion() -> str:
    URL = "https://arxiv.org/category_taxonomy"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="category_taxonomy_list")
    elements = results.find_all("div", class_="columns divided")
    query = ""
    for i, e in enumerate(elements):
        code = e.find("h4").text
        title = e.find("span").text.strip()
        code = code.replace(title, "").strip()
        title = title.replace("(", "").replace(")", "")
        desc = sanitize(e.find("p").text).strip()
        query += f"""
            CREATE (category_{i}:Category {{code: "{code}", title: "{title}", description: "{desc}"}})
        """

    query += f"""
        CREATE (category_astro_ph:Category {{code: "astro-ph", title: "General Astrophysics", description: "General Astrophysics"}})
    """
    return query


def sanitize(text):
    text = (
        str(text)
        .replace("'", "")
        .replace('"', "")
        .replace("{", "")
        .replace("}", "")
        .replace("\n", " ")
        .replace("\\u", "u")
    )
    return text


def create_indices_queries() -> List[str]:
    return [
        "CREATE TEXT INDEX category_code IF NOT EXISTS FOR (c:Category) ON (c.code)",
        "CREATE TEXT INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
        "CREATE TEXT INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)",
    ]


def create_cypher_batch_query_to_insert_arxiv_papers(objs: List[IngestablePaper]):
    items_in_batch = list()
    for o in objs:
        neo4j_date_string = f'date("{o.published_date.strftime("%Y-%m-%d")}")'
        categories_neo4j = '["' + ('","').join(o.categories) + '"]'
        authors_neo4j = '["' + ('","').join(o.authors) + '"]'
        cited_papers_neo4j = '["' + ('","').join(o.cited_arxiv_papers) + '"]'
        batch_string = f'{{id: "{o.arxiv_id}", title: "{sanitize(o.title)}", summary: "{sanitize(o.summary)}", published: {neo4j_date_string}, arxiv_link: "{o.arxiv_link}", pdf_link: "{o.pdf_link}", categories: {categories_neo4j}, authors: {authors_neo4j}, cited_arxiv_papers: {cited_papers_neo4j}}}'
        items_in_batch.append(batch_string)
    query = r"""
    UNWIND $unwind_string as item
    MERGE (paper:Paper {id: item.id})
    ON CREATE
      SET
        paper.title = item.title,
        paper.summary = item.summary,
        paper.published = item.published,
        paper.arxiv_link = item.arxiv_link,
        paper.pdf_link = item.pdf_link,
        paper.cited_arxiv_papers = item.cited_arxiv_papers
    FOREACH (category in item.categories | MERGE (c:Category {code: category}) MERGE (paper)-[:BELONGS_TO_CATEGORY]->(c))
    FOREACH (author in item.authors | MERGE (a:Author {name: author}) MERGE (paper)-[:AUTHORED_BY]->(a))
    """.replace(
        "$unwind_string", "[" + ",".join(items_in_batch) + "]"
    )
    return query


def create_cypher_batch_query_to_create_citation_relationship(arxiv_paper_id: str):
    query = r"""
    MATCH (p:Paper {id: "$arxiv_paper_id"})
    MATCH (cited_papers:Paper) WHERE cited_papers.id IN p.cited_arxiv_papers
    MERGE (p)-[:CITES]->(cited_papers)
    """.replace(
        "$arxiv_paper_id", arxiv_paper_id
    )
    return query
