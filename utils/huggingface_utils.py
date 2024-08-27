import torch
import transformers
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import utils.constants as const

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print(torch.cuda.get_device_name(0))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def quantise_and_save_local_model():
    model = AutoModelForCausalLM.from_pretrained(
        const.local_model_to_be_quantised,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.save_pretrained(save_directory=const.MODELS_PATH)


def load_local_model() -> BaseLLM:
    model = AutoModelForCausalLM.from_pretrained(
        const.MODELS_PATH, trust_remote_code=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(const.local_model_to_be_quantised)
    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=2048,
        temperature=const.llm_temperture,
        do_sample=True,
    )
    local_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return local_llm


def cache_and_load_embedding_model() -> Embeddings:
    embedding = SentenceTransformerEmbeddings(
        model_name=const.embed_model_name,
        cache_folder=const.EMBED_PATH,
        model_kwargs={"trust_remote_code": True},
    )
    return embedding
