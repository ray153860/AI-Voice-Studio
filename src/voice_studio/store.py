import json
import sqlite3
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path=path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            db.execute("CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
    def connect(self):
        db=sqlite3.connect(self.path); db.row_factory=sqlite3.Row; return db
    def save_job(self, job):
        with self.connect() as db: db.execute("INSERT OR REPLACE INTO jobs(id,data) VALUES (?,?)",(job["id"],json.dumps(job)))
    def get_job(self, jid):
        with self.connect() as db:
            row=db.execute("SELECT data FROM jobs WHERE id=?",(jid,)).fetchone()
        return json.loads(row[0]) if row else None
    def list_jobs(self):
        with self.connect() as db: return [json.loads(r[0]) for r in db.execute("SELECT data FROM jobs ORDER BY rowid DESC LIMIT 100")]

    def recover_incomplete_jobs(self):
        recovered = []
        for job in self.list_jobs():
            if job["status"] in ("queued", "running"):
                job["status"] = "failed"
                job["error"] = "Application stopped before the task completed"
                job.setdefault("log", []).append("marked failed during startup recovery")
                self.save_job(job)
                recovered.append(job["id"])
        return recovered
    def delete_job(self, jid):
        with self.connect() as db: db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    def create_project(self, pid, name):
        with self.connect() as db: db.execute("INSERT INTO projects(id,name) VALUES (?,?)",(pid,name))
    def list_projects(self):
        with self.connect() as db: return [dict(r) for r in db.execute("SELECT * FROM projects ORDER BY created_at DESC")]

    def save_model(self, model):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO models(id,data) VALUES (?,?)", (model["id"], json.dumps(model)))

    def get_model(self, model_id):
        with self.connect() as db:
            row = db.execute("SELECT data FROM models WHERE id=?", (model_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_models(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM models ORDER BY rowid DESC")]

    def save_dataset(self, dataset):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO datasets(id,data) VALUES (?,?)", (dataset["id"], json.dumps(dataset)))

    def get_dataset(self, dataset_id):
        with self.connect() as db:
            row = db.execute("SELECT data FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_datasets(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM datasets ORDER BY rowid DESC")]
