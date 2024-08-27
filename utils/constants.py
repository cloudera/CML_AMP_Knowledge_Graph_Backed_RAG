import os

seed_arxiv_paper_ids = [
    "1910.01108",
    # https://github.com/aimerou/awesome-ai-papers?tab=readme-ov-file
    "2302.04761",
    "1706.03762",
    "2312.05934",
    "2302.13971",
    "2112.01488",
    "2305.14314",
    "2303.17564",
    "2310.06825",
    "2304.02643",
    "2301.11305",
    "2304.03277",
    "2304.03442",
    "2310.12931",
    "2309.00267",
    "2212.09720",
    "2404.14047",
    # https://www.zeta-alpha.com/post/trends-in-ai-may-2024
    "2404.11018",
    "2404.07143",
    "2404.10102",
    "2404.10301",
    "2404.18796",
    "2404.05961",
    "2404.18424",
    "2404.02489",
    "2404.19737",
    # https://github.com/dair-ai/ML-Papers-of-the-Week/tree/main?tab=readme-ov-file#top-ml-papers-of-the-week-july-8---july-14---2024
    "2407.02678",
    "2407.07061",
    "2407.02485",
    "2407.04153",
    "2405.18414",
    # https://github.com/dair-ai/ML-Papers-of-the-Week/tree/main?tab=readme-ov-file#top-ml-papers-of-the-week-august-12---august-18---2024
    "2408.04948",
    "2408.04259",
    "2408.05109",
    "2408.02479",
    "2408.02666",
    "2408.02666",
    "2408.01129",
]

EMBED_PATH = "./embed_models"
MODELS_PATH = "./models"
TEMP_VISUAL_GRAPH_PATH = "./temp-graph.html"
TEMP_VISUAL_1_2_GRAPH_PATH = "./temp-first-and-second-order-graph.html"

huggingface_token = os.getenv("HF_TOKEN")

embed_model_name = "Alibaba-NLP/gte-large-en-v1.5"
colbert_model = "colbert-ir/colbertv2.0"
local_model_to_be_quantised = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llm_temperture = 0.01

llama3_stop_token = "<|eot_id|>"
llama3_bos_token = "<|begin_of_text|>"  # Beggining of sequence token
