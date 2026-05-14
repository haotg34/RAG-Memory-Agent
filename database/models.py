from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from datetime import datetime
from database.session import Base


class SessionMemory(Base):
    __tablename__ = "session_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), index=True, nullable=False)
    session_id = Column(String(50), index=True, nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), index=True, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), unique=True, index=True, nullable=False)
    preferences = Column(JSON, default=dict, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_name = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=False)
    chunk_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class LLMUsageLog(Base):
    __tablename__ = "llm_usage_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), index=True, nullable=False)
    session_id = Column(String(50), index=True, nullable=False)
    provider = Column(String(50), index=True, nullable=False)
    model = Column(String(255), index=True, nullable=False)
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class LLMRouteLog(Base):
    __tablename__ = "llm_route_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), index=True, nullable=False)
    session_id = Column(String(50), index=True, nullable=False)
    query = Column(Text, nullable=False)

    decided_tier = Column(String(20), index=True, nullable=False)
    decided_score = Column(String(32), nullable=True)
    rule_hit = Column(String(64), nullable=True)
    checker_raw = Column(String(64), nullable=True)
    upgraded = Column(Integer, default=0, nullable=False)
    degraded = Column(Integer, default=0, nullable=False)

    final_provider = Column(String(50), index=True, nullable=False)
    final_model = Column(String(255), index=True, nullable=False)
    meta = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
