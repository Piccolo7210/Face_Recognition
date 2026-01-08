CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS persons (
    person_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    national_id TEXT UNIQUE,
    embedding VECTOR(128) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS persons_embedding_hnsw
ON persons
USING hnsw (embedding vector_cosine_ops);
