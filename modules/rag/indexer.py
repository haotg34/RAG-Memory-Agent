from sqlalchemy.orm import Session
from database.models import DocumentChunk
from config import settings
from modules.rag.loader import load_document, split_documents
from modules.embedding_client import embed_text
import os

_emb_api_key, _emb_base_url, _emb_model = settings.resolve_embedding()


def index_document(file_path: str, db: Session) -> int:
    document_name = os.path.basename(file_path)

    documents = load_document(file_path)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("文档解析为空，无法索引（请检查是否为扫描件/空文档/复杂排版）")

    chunk_count = 0
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk.page_content, _emb_model, _emb_api_key, _emb_base_url)
        document_chunk = DocumentChunk(
            document_name=document_name,
            chunk_index=i,
            content=chunk.page_content,
            embedding=embedding,
            chunk_metadata=chunk.metadata,
        )
        db.add(document_chunk)
        chunk_count += 1

    db.commit()
    return chunk_count
