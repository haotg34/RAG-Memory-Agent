from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    llm_provider: str = "openai"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-3.5-turbo-0125"
    llm_small_model: str | None = None
    llm_medium_model: str | None = None
    llm_large_model: str | None = None

    routing_enabled: bool = True
    routing_simple_len: int = 12
    routing_score_threshold_small: float = 0.35
    routing_score_threshold_medium: float = 0.7

    llm_timeout_small_s: float = 8.0
    llm_timeout_medium_s: float = 20.0
    llm_timeout_large_s: float = 40.0

    dashscope_api_key: str | None = None
    volc_ark_api_key: str | None = None

    embedding_provider: str = "openai"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"

    dashscope_embedding_api_key: str | None = None
    volc_ark_embedding_api_key: str | None = None

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    database_url_override: str | None = None

    mysql_host: str | None = None
    mysql_port: int = 3306
    mysql_user: str | None = None
    mysql_password: str | None = None
    mysql_database: str | None = None

    postgres_host: str | None = None
    postgres_port: int = 5432
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_database: str | None = None

    max_tokens: int = 4096
    temperature: float = 0.0
    memory_window_size: int = 10
    long_term_memory_top_k: int = 3
    rag_top_k: int = 5
    rag_semantic_top_k: int = 40
    rag_keyword_top_k: int = 40
    rag_candidate_top_k: int = 30

    def resolve_llm_for(self, provider: str | None = None) -> tuple[str, str, str]:
        p = (provider or self.llm_provider or "openai").lower()
        base_url = (
            self.llm_base_url
            if provider is None or p == (self.llm_provider or "").lower()
            else None
        )
        base_url = base_url or self._provider_base_url(p, self.openai_base_url)
        model = self.llm_model

        if p == "dashscope":
            api_key = self.llm_api_key or self.dashscope_api_key or ""
        elif p in ("volc_ark", "volc_ark_coding"):
            api_key = self.volc_ark_api_key or ""
        elif p == "openai":
            api_key = self.openai_api_key or self.llm_api_key or ""
        else:
            api_key = self.llm_api_key or self.openai_api_key or ""

        if provider is not None and p in ("volc_ark", "volc_ark_coding") and not api_key:
            raise ValueError("未配置方舟 API Key：请在 .env 设置 VOLC_ARK_API_KEY")
        if provider is not None and p == "dashscope" and not api_key:
            raise ValueError("未配置百炼 API Key：请在 .env 设置 LLM_API_KEY 或 DASHSCOPE_API_KEY")
        if provider is not None and p == "openai" and not api_key:
            raise ValueError("未配置 OpenAI API Key：请在 .env 设置 OPENAI_API_KEY")

        return api_key, base_url, model

    def resolve_llm(self) -> tuple[str, str, str]:
        return self.resolve_llm_for(None)

    def resolve_embedding(self) -> tuple[str, str, str]:
        provider = (self.embedding_provider or "openai").lower()
        if provider == "dashscope":
            api_key = self.embedding_api_key or self.dashscope_embedding_api_key or self.dashscope_api_key or ""
        elif provider in ("volc_ark", "volc_ark_coding"):
            api_key = self.embedding_api_key or self.volc_ark_embedding_api_key or self.volc_ark_api_key or ""
        else:
            api_key = self.embedding_api_key or self.openai_api_key or ""
        base_url = self.embedding_base_url or self._provider_base_url(provider, self.openai_base_url)
        model = self.embedding_model
        return api_key, base_url, model

    def _provider_base_url(self, provider: str, fallback: str) -> str:
        presets = {
            "openai": fallback,
            "volc_ark": "https://ark.cn-beijing.volces.com/api/v3",
            "volc_ark_coding": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        return presets.get(provider, fallback)

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        if self.postgres_host and self.postgres_user and self.postgres_password and self.postgres_database:
            return (
                f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            )
        if self.mysql_host and self.mysql_user and self.mysql_password and self.mysql_database:
            return (
                f"mysql+mysqlconnector://{self.mysql_user}:{self.mysql_password}"
                f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            )
        raise ValueError("未配置数据库连接：请设置 DATABASE_URL_OVERRIDE 或提供 MySQL/PostgreSQL 配置")


settings = Settings()
