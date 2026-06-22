PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
  project_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
  profile_id TEXT PRIMARY KEY,
  mode TEXT NOT NULL,
  path TEXT NOT NULL,
  profile_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
  hypothesis_id TEXT PRIMARY KEY,
  title TEXT,
  summary TEXT,
  created_at TEXT,
  model TEXT,
  reasoning_tokens INTEGER,
  total_tokens INTEGER,
  generation_seconds INTEGER,
  enabled INTEGER NOT NULL,
  path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS materializations (
  hypothesis_id TEXT NOT NULL,
  mode TEXT NOT NULL,
  file TEXT NOT NULL,
  code_hash TEXT NOT NULL,
  model TEXT,
  reasoning_tokens INTEGER,
  total_tokens INTEGER,
  generation_seconds INTEGER,
  PRIMARY KEY (hypothesis_id, mode, file)
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  step INTEGER,
  hypothesis_id TEXT,
  mode TEXT,
  profile_id TEXT,
  status TEXT NOT NULL,
  created_at TEXT,
  finished_at TEXT,
  run_seconds INTEGER,
  path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
  node_id TEXT PRIMARY KEY,
  hypothesis_id TEXT,
  mode TEXT,
  profile_id TEXT,
  code_hash TEXT,
  metric REAL,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  node_id TEXT NOT NULL,
  path TEXT NOT NULL,
  kind TEXT NOT NULL,
  PRIMARY KEY (node_id, path)
);

CREATE TABLE IF NOT EXISTS submissions (
  node_id TEXT NOT NULL,
  submission_path TEXT NOT NULL,
  submission_sha256 TEXT NOT NULL,
  submission_size INTEGER NOT NULL,
  submission_mtime_ns INTEGER NOT NULL,
  run_id TEXT,
  step INTEGER,
  hypothesis_id TEXT,
  mode TEXT,
  profile_id TEXT,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  submit_status TEXT NOT NULL,
  local_score REAL,
  public_score REAL,
  public_rank INTEGER,
  metric TEXT,
  code_hash TEXT,
  run_seconds INTEGER,
  created_at TEXT,
  finished_at TEXT,
  artifact_dir TEXT NOT NULL,
  submitted_at TEXT,
  kaggle_message TEXT,
  kaggle_response_json TEXT,
  kaggle_ref TEXT,
  upload_path TEXT,
  uploaded_filename TEXT,
  remote_status TEXT,
  remote_date TEXT,
  remote_url TEXT,
  private_score REAL,
  PRIMARY KEY (node_id, submission_path)
);

CREATE INDEX IF NOT EXISTS idx_submissions_sha256 ON submissions(submission_sha256);
CREATE INDEX IF NOT EXISTS idx_submissions_local_score ON submissions(local_score);
CREATE INDEX IF NOT EXISTS idx_submissions_public_score ON submissions(public_score);

CREATE TABLE IF NOT EXISTS prompt_calls (
  call_id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  template_id TEXT,
  rendered_prompt_hash TEXT
);
