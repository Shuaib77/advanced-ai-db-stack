from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache

def initialize_caching():
    # Route all LLM requests through the Redis cache first
    set_llm_cache(RedisSemanticCache(
        redis_url="redis://localhost:6379",
        embedding=OpenAIEmbeddings()
    ))
    print("Semantic Cache Active: Similar prompts will bypass the LLM API.")
