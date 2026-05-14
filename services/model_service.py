import time
from openai import OpenAI
from config import settings

_CACHE: dict[tuple[str, str], tuple[float, list[str]]] = {}


def list_models(provider: str) -> list[str]:
    p = (provider or "").lower()
    api_key, base_url, _ = settings.resolve_llm_for(p)
    key = (p, base_url)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < 300:
        return cached[1]

    client = OpenAI(api_key=api_key, base_url=base_url)
    r = client.models.list()
    models = [m.id for m in getattr(r, "data", []) if getattr(m, "id", None)]
    models = sorted(set(models))
    _CACHE[key] = (now, models)
    return models
