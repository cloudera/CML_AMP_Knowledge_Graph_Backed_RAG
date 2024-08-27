import utils.constants as const
from utils.huggingface_utils import (
    cache_and_load_embedding_model,
    quantise_and_save_local_model,
)

# This just caches the embedding model for future use
cache_and_load_embedding_model()

# cache the Llama 3 8b local model in a 4-bit quantised format
quantise_and_save_local_model()
