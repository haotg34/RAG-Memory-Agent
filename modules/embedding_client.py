from functools import lru_cache
from openai import OpenAI


@lru_cache(maxsize=4)
def _client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def embed_text(text: str, model: str, api_key: str, base_url: str) -> list[float]:
    r = _client(api_key, base_url).embeddings.create(model=model, input=text)
    return r.data[0].embedding


def embed_texts(texts: list[str], model: str, api_key: str, base_url: str) -> list[list[float]]:
    r = _client(api_key, base_url).embeddings.create(model=model, input=texts)
    return [d.embedding for d in r.data]
