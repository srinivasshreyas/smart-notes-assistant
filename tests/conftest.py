import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A fresh app with its own temporary database, so tests never touch data/notes.db."""
    monkeypatch.setenv("SMART_NOTES_DB", str(tmp_path / "test_notes.db"))
    from app.main import app
    with TestClient(app) as c:  # 'with' runs the startup (model loading)
        yield c
