from typing import List, Dict
from langchain.graphs import Neo4jGraph
import networkx as nx
from pyvis.network import Network
import streamlit as st

import utils.constants as const

def _get_raw_auxillary_context_for_papers(paper_ids: List[str], graphDbInstance: Neo4jGraph):
    query = r"""
    MATCH (p:Paper)
    WHERE p.id in $list
    CALL {
        WITH p
        MATCH (p)-[:AUTHORED_BY]->(a:Author)
        WITH a, COUNT {(a)<-[:AUTHORED_BY]-(:Paper)} AS paper_written_by_author_count
        ORDER BY paper_written_by_author_count DESC
        LIMIT 3
        RETURN a
    }
    CALL {
        WITH p
        MATCH (p)<-[:CITES]-(top_paper:Paper)
        WITH top_paper, COUNT {(top_paper)<-[:CITES]-(:Paper)} AS papers_citing_top_paper_count
        ORDER BY papers_citing_top_paper_count DESC
        LIMIT 3
        RETURN top_paper
    }
    RETURN p, top_paper, a
    """.replace("$list", str(paper_ids))
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
    """.replace("$list", str(arxiv_ids))
    results = graphDbInstance.query(query)
    return results[0]["node_pairs"]

def _create_networkx_graph(paper_ids: List[str], graphDbInstance: Neo4jGraph):
    def _get_hover_data(paper: Dict):
        hover_string = paper['title'] + "\n"
        hover_string += "Arxiv ID: " + paper['id'] + "\n"
        hover_string += "Published: " + paper['published'].to_native().strftime("%B %d, %Y")
        return hover_string

    data = _get_raw_auxillary_context_for_papers(paper_ids, graphDbInstance)
    unique_papers = set()
    G = nx.DiGraph()
    for record in data:
        top_paper = record['top_paper']
        unique_papers.add(top_paper['id'])
        G.add_node(top_paper['id'], label=top_paper['title'], color='violet', title=_get_hover_data(top_paper), node_type='Paper')
    for record in data:
        p, author = record['p'], record['a']
        unique_papers.add(p['id'])
        G.add_node(p['id'], label=p['title'], color='blue', title=_get_hover_data(p), node_type='Paper')
        G.add_node(author['name'], label=author['name'], color='orange', node_type='Author')
        G.add_edges_from([
            (p['id'], author['name'], {'label': 'AUTHORED_BY'}),
        ])
    node_pairs = _get_citation_relationships(list(unique_papers), graphDbInstance)
    for pair in node_pairs:
        G.add_edges_from([
            (pair[0], pair[1], {'label': 'CITES'}),
        ])
    return G

def visualize_graph(paper_ids: List[str], graphDbInstance: Neo4jGraph):
    progress_bar = st.progress(20, "Running cypher query for auxiliary context.")
    G = _create_networkx_graph(paper_ids, graphDbInstance)
    progress_bar.progress(60, "Rendering graph.")
    net = Network(notebook=True)
    net.from_nx(G)
    progress_bar.progress(90, "Saving graph.")
    net.show(const.TEMP_VISUAL_GRAPH_PATH)
    # make the paper nodes clickable
    with open(const.TEMP_VISUAL_GRAPH_PATH, "r") as f:
        html_content = f.read()
        content_to_be_added = """
        network.on( 'click', function(properties) {
            var ids = properties.nodes;
            var clickedNode = nodes.get(ids)[0];
            if (clickedNode.node_type == "Paper") {
                window.open('https://arxiv.org/abs/'+clickedNode.id, '_blank');
            }
        });
        """
        string_to_find = "vis.Network(container, data, options);"
        html_content = html_content.replace(string_to_find, string_to_find + content_to_be_added)
    with open(const.TEMP_VISUAL_GRAPH_PATH, "w") as f:
        f.write(html_content)
    progress_bar.empty()
