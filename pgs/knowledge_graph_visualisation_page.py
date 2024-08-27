from typing import Dict, List

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from langchain.graphs import Neo4jGraph
from pyvis.network import Network

import utils.constants as const
from utils.neo4j_utils import (
    get_neo4j_credentails,
    is_neo4j_server_up,
    wait_for_neo4j_server,
)

with st.spinner("Spinning up the Neo4j server..."):
    if not is_neo4j_server_up():
        wait_for_neo4j_server()

    graph = Neo4jGraph(
        username=get_neo4j_credentails()["username"],
        password=get_neo4j_credentails()["password"],
        url=get_neo4j_credentails()["uri"],
    )


def _get_all_papers(graphDbInstance: Neo4jGraph):
    query = r"""
    MATCH (p:Paper)
    WITH p, COUNT {(p)<-[:CITES]-(:Paper)} AS citation_count
    ORDER BY citation_count DESC
    RETURN p, citation_count
    """
    results = graphDbInstance.query(query)
    return results


def _get_first_and_second_order_citing_papers(
    arxiv_id: str, graphDbInstance: Neo4jGraph
):
    query = r"""
    MATCH (p:Paper {id: "$arxiv_id"})
    RETURN {
        paper: p,
        p1s: COLLECT {MATCH (p)<--(p1:Paper) RETURN p1},
        p2s: COLLECT {MATCH (p)<--(:Paper)<--(p2:Paper) RETURN p2}
    } AS result
    """.replace(
        "$arxiv_id", arxiv_id
    )
    results = graphDbInstance.query(query)
    return results


def _get_citation_relationships(arxiv_ids: List[str], graphDbInstance: Neo4jGraph):
    query = r"""
    MATCH (p:Paper)
    WHERE p.id in $list
    WITH COLLECT(ELEMENTID(p)) as paper_ids
    CALL apoc.algo.cover(paper_ids)
    YIELD rel
    WITH COLLECT(rel) AS citations
    RETURN [r IN citations | [startNode(r).id, endNode(r).id]] AS node_pairs
    """.replace(
        "$list", str(arxiv_ids)
    )
    results = graphDbInstance.query(query)
    return results[0]["node_pairs"]


def _create_knowledege_base_networkX_graph(
    arxiv_id: str, graphDbInstance: Neo4jGraph
) -> nx.Graph:
    def _get_hover_data(paper: Dict):
        hover_string = paper["title"] + "\n"
        hover_string += "Arxiv ID: " + paper["id"] + "\n"
        hover_string += "Published: " + paper["published"].to_native().strftime(
            "%B %d, %Y"
        )
        return hover_string

    data = _get_first_and_second_order_citing_papers(arxiv_id, graphDbInstance)
    unique_papers = set()
    paper = data[0]["result"]["paper"]
    p1s = data[0]["result"]["p1s"]
    p2s = data[0]["result"]["p2s"]
    G = nx.DiGraph()
    G.add_node(
        paper["id"],
        label=paper["title"],
        color="blue",
        title=_get_hover_data(paper),
        node_type="Paper",
    )
    unique_papers.add(paper["id"])
    for p1 in p1s:
        if p1["id"] in unique_papers:
            continue
        G.add_node(
            p1["id"],
            label=p1["title"],
            color="violet",
            title=_get_hover_data(p1),
            node_type="Paper",
        )
        unique_papers.add(p1["id"])
    for p2 in p2s:
        if p2["id"] in unique_papers:
            continue
        G.add_node(
            p2["id"],
            label=p2["title"],
            color="green",
            title=_get_hover_data(p2),
            node_type="Paper",
        )
        unique_papers.add(p2["id"])
    node_pairs = _get_citation_relationships(list(unique_papers), graphDbInstance)
    for pair in node_pairs:
        G.add_edges_from(
            [
                (pair[0], pair[1], {"label": "CITES"}),
            ]
        )
    return G


def visualise_first_and_second_degree_cited_by_papers(
    arxiv_id: str, graphDbInstance: Neo4jGraph
):
    G = _create_knowledege_base_networkX_graph(arxiv_id, graphDbInstance)
    net = Network(notebook=True)
    net.from_nx(G)
    net.show(const.TEMP_VISUAL_1_2_GRAPH_PATH)
    with open(const.TEMP_VISUAL_1_2_GRAPH_PATH, "r") as f:
        html_content = f.read()
        content_to_be_added = """
        network.on( 'click', function(properties) {
            var ids = properties.nodes;
            var clickedNode = nodes.get(ids)[0];
            if (clickedNode.node_type == "Paper") {
                spanId = "paper-entry-" + clickedNode.id;
                parent.document.getElementById(spanId).scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });
            }
        });
        """
        string_to_find = "vis.Network(container, data, options);"
        html_content = html_content.replace(
            string_to_find, string_to_find + content_to_be_added
        )
    with open(const.TEMP_VISUAL_1_2_GRAPH_PATH, "w") as f:
        f.write(html_content)


paper_col, viz_col = st.columns([0.4, 0.6], gap="small")
paper_col.markdown("## :blue[_arXiv_] papers in the Knowledge Graph")
paper_container = paper_col.container(height=700, border=False)
graph_header = viz_col.container(border=False)
graph_container = viz_col.container(height=600, border=False)


def button_callback(arxiv_id: str):
    graph_header.markdown("## Knowledge Graph Visualization")
    visualise_first_and_second_degree_cited_by_papers(arxiv_id, graph)
    htmlfile = open(const.TEMP_VISUAL_1_2_GRAPH_PATH, "r", encoding="utf-8")
    htmlfile_source_code = htmlfile.read()
    graph_container.empty()
    with graph_container:
        components.html(htmlfile_source_code, height=570, scrolling=True)
    viz_col.markdown(
        f"""
Showing first and second degree \"cited by\" relationships for paper #[{arxiv_id}](https://arxiv.org/abs/{arxiv_id})
 - :blue[Blue] : The selected paper
 - :violet[Violet] : Papers that cite the selected paper
 - :green[Green] : Papers that cite the papers that cite the selected paper
"""
    )


all_papers_data = _get_all_papers(graph)
for record in all_papers_data:
    paper = record["p"]
    citation_count = record["citation_count"]
    arxiv_id = paper["id"]
    arxiv_link = paper["arxiv_link"]
    published_string = paper["published"].to_native().strftime("%B %d, %Y")
    paper_title = paper["title"]
    sub_container = paper_container.container(border=False)
    sub_container.markdown(
        f"""
<span id="paper-entry-{arxiv_id}"></span>
#### {paper_title}
**Arxiv ID**: [{arxiv_id}]({arxiv_link})  
**Published On**: {published_string}  
**Citiation Count**: {citation_count}
""",
        unsafe_allow_html=True,
    )
    sub_container.button(
        "Visualize as Knowledge Graph",
        key="button--" + arxiv_id,
        on_click=button_callback,
        args=(arxiv_id,),
    )
    sub_container.markdown("---")
