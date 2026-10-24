"""SQLite schema for immutable message sources and ingestion journals."""

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
INSERT INTO schema_version(version)
SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    created_at TEXT NOT NULL,
    input_hash TEXT,
    content_hash TEXT NOT NULL,
    UNIQUE(session_id, turn_id, role)
);

CREATE TABLE IF NOT EXISTS message_parts (
    message_id TEXT NOT NULL,
    part_index INTEGER NOT NULL,
    part_type TEXT NOT NULL CHECK(part_type IN ('text', 'image')),
    text_content TEXT,
    image_bytes BLOB,
    media_type TEXT,
    content_hash TEXT NOT NULL,
    source_metadata_json TEXT NOT NULL,
    PRIMARY KEY(message_id, part_index),
    FOREIGN KEY(message_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    result_json TEXT,
    error_json TEXT,
    checkpoint_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_turn_state (
    session_id TEXT PRIMARY KEY,
    next_turn INTEGER NOT NULL CHECK(next_turn >= 1)
);

CREATE TABLE IF NOT EXISTS session_submissions (
    session_id TEXT NOT NULL,
    submission_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('processing', 'completed', 'failed')),
    message_ids_json TEXT NOT NULL,
    ingestion_run_id TEXT,
    worker_token TEXT,
    lease_expires_at TEXT,
    result_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(session_id, submission_id),
    FOREIGN KEY(ingestion_run_id) REFERENCES ingestion_runs(id)
);

CREATE TABLE IF NOT EXISTS agent_exchanges (
    id TEXT PRIMARY KEY,
    request_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    user_request_hash TEXT NOT NULL,
    assistant_request_hash TEXT,
    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    stage TEXT NOT NULL,
    user_message_id TEXT NOT NULL,
    assistant_message_id TEXT,
    before_result_json TEXT,
    after_result_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_message_id) REFERENCES messages(id),
    FOREIGN KEY(assistant_message_id) REFERENCES messages(id)
);

CREATE INDEX IF NOT EXISTS idx_agent_exchanges_session
ON agent_exchanges(session_id, created_at);

CREATE TABLE IF NOT EXISTS ingestion_candidates (
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    candidate_json TEXT NOT NULL,
    decision_json TEXT,
    status TEXT NOT NULL,
    PRIMARY KEY(run_id, candidate_id),
    FOREIGN KEY(run_id) REFERENCES ingestion_runs(id)
);

CREATE TABLE IF NOT EXISTS ingestion_write_lock (
    lock_name TEXT PRIMARY KEY,
    owner_operation_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,
    version INTEGER NOT NULL,
    summarized_through_sequence INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    request_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_submissions (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    ingestion_run_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    PRIMARY KEY(session_id, sequence),
    FOREIGN KEY(ingestion_run_id) REFERENCES ingestion_runs(id)
);

CREATE TABLE IF NOT EXISTS summary_submission_messages (
    session_id TEXT NOT NULL,
    submission_sequence INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    PRIMARY KEY(session_id, submission_sequence, ordinal),
    FOREIGN KEY(session_id, submission_sequence)
        REFERENCES summary_submissions(session_id, sequence),
    FOREIGN KEY(message_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS summary_jobs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    from_sequence INTEGER NOT NULL,
    through_sequence INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'complete', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    error_json TEXT,
    available_at TEXT NOT NULL,
    claimed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, from_sequence, through_sequence)
);

CREATE INDEX IF NOT EXISTS idx_summary_jobs_ready
ON summary_jobs(status, available_at, created_at);

CREATE INDEX IF NOT EXISTS idx_summary_submissions_session
ON summary_submissions(session_id, sequence);

UPDATE schema_version SET version = 4 WHERE version < 4;
"""
