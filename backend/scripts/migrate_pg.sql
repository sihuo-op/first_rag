-- Chunk 生命周期扩容迁移脚本
-- 执行方式：sqlite3 data/sqlite/app.db < scripts/migrate_pg.sql
-- 或在 PG 客户端中执行

-- ============ document_chunks 表 ============
ALTER TABLE document_chunks ADD COLUMN content_hash VARCHAR(64);
ALTER TABLE document_chunks ADD COLUMN status VARCHAR(20) DEFAULT 'active' NOT NULL;
ALTER TABLE document_chunks ADD COLUMN conflict_with_chunk_id VARCHAR(100);
ALTER TABLE document_chunks ADD COLUMN conflict_detected_at TIMESTAMP;
ALTER TABLE document_chunks ADD COLUMN confidence FLOAT;
ALTER TABLE document_chunks ADD COLUMN review_reason TEXT;
ALTER TABLE document_chunks ADD COLUMN superseded_at TIMESTAMP;
ALTER TABLE document_chunks ADD COLUMN reviewed_by INTEGER REFERENCES users(id);
ALTER TABLE document_chunks ADD COLUMN reviewed_at TIMESTAMP;
ALTER TABLE document_chunks ADD COLUMN access_count INTEGER DEFAULT 0 NOT NULL;
ALTER TABLE document_chunks ADD COLUMN last_accessed_at TIMESTAMP;
ALTER TABLE document_chunks ADD COLUMN hit_count INTEGER DEFAULT 0 NOT NULL;
ALTER TABLE document_chunks ADD COLUMN total_score FLOAT DEFAULT 0.0 NOT NULL;
ALTER TABLE document_chunks ADD COLUMN avg_score FLOAT DEFAULT 0.0 NOT NULL;
ALTER TABLE document_chunks ADD COLUMN archived_reason VARCHAR(30);
ALTER TABLE document_chunks ADD COLUMN archived_at TIMESTAMP;

CREATE INDEX idx_document_chunks_milvus_id ON document_chunks(milvus_id);
CREATE INDEX idx_document_chunks_status ON document_chunks(status);

-- ============ documents 表 ============
ALTER TABLE documents ADD COLUMN conflict_check_status VARCHAR(20) DEFAULT 'completed' NOT NULL;
ALTER TABLE documents ADD COLUMN conflict_check_started_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN conflict_check_completed_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN conflict_check_progress VARCHAR(20);
