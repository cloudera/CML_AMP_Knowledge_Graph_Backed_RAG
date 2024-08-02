import logging
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate

import utils.retriever_utils as ret_utils
import utils.constants as const

class VanillaRAG:
    _prompt_template= """<|start_header_id|>system<|end_header_id|>
You are an AI language model designed to assist with retrieval-augmented generation tasks. Your job is to provide detailed, accurate, and contextually relevant information by leveraging external knowledge sources.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Use natural language and be concise.
Return the Document arXiv ID for the Document used to answer the question. Return the arXiv ID between square brackets prefixed with "arXiv:". For example, if the arXiv ID is 1234.56789, return [arXiv:1234.56789].
If multiple documents are used to generate the answer, return the arXiv IDs separated by commas. For example, if the arXiv IDs are 1234.56789, 6711.03217 and 4512.09876, return [arXiv:1234.56789, arXiv:6711.03217, arXiv:4512.09876].
First state the arXiv ID and then answer the question.
Say that you don't know the answer if you don't find it in the context.
Answer the question based only on the following context.:
Context: {context}

Question: {question}
Try to combine information from multiple documents to answer the question.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def __init__(self, graphDbInstance: Neo4jGraph, document_index: Neo4jVector, llm: BaseLLM, top_k: int, bos_token: str):
        self.graphDbInstance = graphDbInstance
        self.document_index = document_index
        self.llm = llm
        self.top_k = top_k
        self.bos_token = bos_token

    def retrieve_context(self, query: str) -> str:
        paper_chunks = ret_utils.vanilla_retreiver(query=query, top_k=self.top_k, graphDbInstance=self.graphDbInstance, document_index=self.document_index)
        context = ""
        for chunk in paper_chunks:
            context += f"Document:{chunk.text}\n"
            context += f"Document arXiv ID: {chunk.paper.arxiv_id}\n"
            context += "\n\n"
        return context
    
    def invoke(self, question: str) -> str:
        context = self.retrieve_context(question)
        logging.debug(f"Context: {context}")
        prompt1 = PromptTemplate.from_template(self.bos_token+self._prompt_template)
        chain1 = prompt1 | self.llm
        response1 = chain1.invoke({
            "question": question,
            "context": context,
        })
        return response1
