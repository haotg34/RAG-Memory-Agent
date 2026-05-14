from sqlalchemy.orm import Session
from database.models import SessionMemory, LongTermMemory, UserPreference
import numpy as np
from config import settings
from modules.memory.extractor import extract_long_term_memory
from modules.embedding_client import embed_text

_emb_api_key, _emb_base_url, _emb_model = settings.resolve_embedding()


class MemoryManager:
    def __init__(self, user_id: str, db: Session):
        self.user_id = user_id
        self.db = db
        self.short_term_memory = []

    def add_message(self, role: str, content: str, session_id: str):
        self.short_term_memory.append({"role": role, "content": content})

        if len(self.short_term_memory) > settings.memory_window_size * 2:
            self.short_term_memory = self.short_term_memory[-settings.memory_window_size * 2 :]

        session_memory = SessionMemory(
            user_id=self.user_id,
            session_id=session_id,
            role=role,
            content=content,
        )
        self.db.add(session_memory)
        self.db.commit()

    def get_recent_memory(self, session_id: str, limit: int | None = None) -> list:
        if limit is None:
            limit = settings.memory_window_size

        memories = (
            self.db.query(SessionMemory)
            .filter(
                SessionMemory.user_id == self.user_id,
                SessionMemory.session_id == session_id,
            )
            .order_by(SessionMemory.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [{"role": m.role, "content": m.content} for m in reversed(memories)]

    def retrieve_long_term_memory(self, query: str, top_k: int | None = None) -> list:
        if top_k is None:
            top_k = settings.long_term_memory_top_k

        query_embedding = embed_text(query, _emb_model, _emb_api_key, _emb_base_url)
        all_memories = (
            self.db.query(LongTermMemory).filter(LongTermMemory.user_id == self.user_id).all()
        )

        if not all_memories:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q) + 1e-12

        results = []
        for memory in all_memories:
            m = np.array(memory.embedding, dtype=np.float32)
            sim = float(np.dot(q, m) / (q_norm * (np.linalg.norm(m) + 1e-12)))
            results.append((memory.content, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return [content for content, sim in results[:top_k] if sim > 0.7]

    def save_long_term_memory(self, conversation: list):
        memories = extract_long_term_memory(conversation)
        if not memories:
            return

        for memory in memories:
            embedding = embed_text(memory, _emb_model, _emb_api_key, _emb_base_url)
            self.db.add(
                LongTermMemory(
                    user_id=self.user_id,
                    content=memory,
                    embedding=embedding,
                )
            )

        self.db.commit()

    def get_user_preferences(self) -> dict:
        preference = (
            self.db.query(UserPreference).filter(UserPreference.user_id == self.user_id).first()
        )
        return preference.preferences if preference else {}

    def update_user_preferences(self, new_preferences: dict):
        if not new_preferences:
            return

        preference = (
            self.db.query(UserPreference).filter(UserPreference.user_id == self.user_id).first()
        )
        if not preference:
            preference = UserPreference(user_id=self.user_id, preferences={})
            self.db.add(preference)

        preference.preferences.update(new_preferences)
        self.db.commit()
