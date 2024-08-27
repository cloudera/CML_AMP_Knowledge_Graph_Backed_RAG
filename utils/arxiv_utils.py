import io
import re
from datetime import date, datetime
from typing import Dict, List

import arxiv
import requests
from langchain.graphs import Neo4jGraph
from PyPDF2 import PdfReader

import utils.constants as const


class IngestablePaper:
    def __init__(
        self,
        arxiv_id: str,
        arxiv_link: str,
        title: str,
        summary: str,
        authors: List[str],
        categories: List[str],
        pdf_link: str,
        published_date: date,
        full_text: str,
        cited_arxiv_papers: List[str],
    ):
        self.arxiv_id = arxiv_id
        self.arxiv_link = arxiv_link
        self.title = title
        self.summary = summary
        self.authors = authors
        self.categories = categories
        self.pdf_link = pdf_link
        self.published_date = published_date
        self.full_text = full_text
        self.cited_arxiv_papers = cited_arxiv_papers
        self._citation_count = None
        self._graph_db_instance = None

    @property
    def citation_count(self):
        return self._citation_count

    @citation_count.setter
    def citation_count(self, value):
        self._citation_count = value

    @property
    def graph_db_instance(self) -> Neo4jGraph:
        return self._graph_db_instance

    @graph_db_instance.setter
    def graph_db_instance(self, value):
        self._graph_db_instance = value

    # the papers are returned in the order of which paper has most citations
    def get_citing_papers(self) -> List["IngestablePaper"]:
        query = r"""MATCH (p:Paper)-[:CITES]->(cited:Paper)
        WHERE cited.id = '$id'
        RETURN {
        id: p.id, title: p.title, summary: p.summary, published: p.published, arxiv_link: p.arxiv_link, pdf_link: p.pdf_link, cited_arxiv_papers: p.cited_arxiv_papers,
        authors: COLLECT { MATCH (p)-->(a:Author) RETURN a.name },
        categories: COLLECT { MATCH (p)-->(c:Category) RETURN c.code },
        citations: COUNT { (p)<-[:CITES]-(:Paper) }
        } AS result
        ORDER BY result.citations DESC
        """.replace(
            "$id", self.arxiv_id
        )
        results = self.graph_db_instance.query(query)
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

    def get_top_authors(self) -> List[str]:
        query = r"""MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
        WHERE p.id='$id'
        WITH DISTINCT a
        RETURN {
            name: a.name,
            paper_ids: COLLECT {MATCH (op:Paper)-[:AUTHORED_BY]->(a:Author) RETURN op.id }
        } AS result
        ORDER BY SIZE(result.paper_ids) DESC
        """.replace(
            "$id", self.arxiv_id
        )
        results = self.graph_db_instance.query(query)
        return [r["result"]["name"] for r in results]


class PaperChunk:
    def __init__(self, text: str, paper: IngestablePaper):
        self.text = text
        self.paper = paper
        self._metadata = dict()

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value) -> Dict:
        self._metadata = value


def extract_pdf_link_from_result(res: arxiv.Result):
    for l in res.links:
        if l.title == "pdf":
            return l.href
    raise ValueError("No PDF link found in the result.")


def convert_pdf_link_to_text(pdf_link: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36"
    }
    response = requests.get(url=pdf_link, headers=headers, timeout=120)
    on_fly_mem_obj = io.BytesIO(response.content)
    pdf_file = PdfReader(on_fly_mem_obj)
    text = ""
    for page in pdf_file.pages:
        text += page.extract_text() + "\n"
    return text


def get_cited_arxiv_papers_from_paper_text(
    original_paper: arxiv.Result, text: str
) -> List[str]:
    pattern = r"arXiv:\d{4}\.\d{4,5}"
    arxiv_references = re.findall(pattern, text)
    arxiv_ids = [arxiv_id.split(":")[1] for arxiv_id in arxiv_references]
    # remove duplicate arxiv ids
    arxiv_ids = list(set(arxiv_ids))
    # remove the original paper's id
    arxiv_ids = [
        arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in original_paper.entry_id
    ]
    return arxiv_ids


def create_paper_object_from_arxiv_id(arxivId: str) -> IngestablePaper:
    client = arxiv.Client()
    itr = client.results(arxiv.Search(id_list=[arxivId]))
    result = next(itr)
    pdf_link = extract_pdf_link_from_result(result)
    full_text = convert_pdf_link_to_text(pdf_link)
    cited_arxiv_papers = get_cited_arxiv_papers_from_paper_text(result, full_text)
    return IngestablePaper(
        arxiv_id=re.findall(r"\d{4}\.\d{4,5}", result.entry_id)[0],
        arxiv_link=result.entry_id,
        title=result.title,
        summary=result.summary,
        authors=[a.name for a in result.authors],
        categories=result.categories,
        pdf_link=pdf_link,
        published_date=result.published.date(),
        full_text=full_text,
        cited_arxiv_papers=cited_arxiv_papers,
    )


def linkify_authors(text: str, authors: List[str]) -> str:
    authors = list(set(authors))
    new_text = text
    for author in authors:
        search_author = author.replace(" ", "+")
        new_text = new_text.replace(
            author,
            f"[{author}](https://arxiv.org/search/cs?query={search_author}&searchtype=author&abstracts=show&order=-announced_date_first&size=50)",
        )
    return new_text


def linkify_arxiv_ids(text: str) -> str:
    arxiv_id_pattern = r"\d{4}\.\d{4,5}"
    arxiv_ids = re.findall(arxiv_id_pattern, text)
    arxiv_ids = list(set(arxiv_ids))
    new_text = text
    for arxiv_id in arxiv_ids:
        new_text = new_text.replace(
            arxiv_id, f"[{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"
        )
    return new_text
