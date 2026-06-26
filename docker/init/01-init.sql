CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_base (
    id VARCHAR(64) PRIMARY KEY,
    content TEXT NOT NULL,
    source VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_embedding_cos
    ON knowledge_base USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id      VARCHAR(32) PRIMARY KEY,
    user_id        VARCHAR(64) NOT NULL,
    type           VARCHAR(32) NOT NULL,
    priority       VARCHAR(16) NOT NULL,
    status         VARCHAR(20) NOT NULL DEFAULT 'created',
    summary        TEXT,
    details        TEXT,
    created_at     TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMP NOT NULL DEFAULT NOW(),
    resolved_at    TIMESTAMP,
    resolve_note   TEXT
);

CREATE INDEX IF NOT EXISTS idx_tickets_user ON tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at DESC);

CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id       VARCHAR(64) PRIMARY KEY,
    title        VARCHAR(255) NOT NULL,
    category     VARCHAR(32) NOT NULL DEFAULT 'general',
    content      TEXT NOT NULL,
    source       VARCHAR(255),
    embedding    VECTOR(1024),
    created_by   VARCHAR(64) DEFAULT 'system',
    created_at   TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_docs_category ON kb_documents(category);
CREATE INDEX IF NOT EXISTS idx_kb_docs_created ON kb_documents(created_at DESC);