"""FastAPI app: POST /classify (fixed contract) + notes, search and a plain HTML page.

Start:  uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictStr, field_validator

from app import db
from app.classifier import Classifier
from app.search import search_tfidf

STATIC = Path(__file__).parent / "static"
MAX_CHARS = 1000
LABELS = {"Study", "Money", "Travel", "Shopping", "Uncertain"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = Classifier()  # loaded ONCE at startup
    db.init()
    yield


app = FastAPI(title="Smart Notes Assistant", lifespan=lifespan)


class NoteIn(BaseModel):
    text: StrictStr  # strict: numbers, null, lists are rejected, not converted

    @field_validator("text")
    @classmethod
    def check(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Note text is empty. Please type something.")
        if len(v) > MAX_CHARS:
            raise ValueError(f"Note is too long (max {MAX_CHARS} characters).")
        return v


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    """Every invalid body (missing field, null, number, empty, bad JSON) becomes a clean 422."""
    return JSONResponse(status_code=422, content={
        "detail": "Invalid request: 'text' must be a non-empty string of at most "
                  f"{MAX_CHARS} characters, sent as JSON like {{\"text\": \"...\"}}."})


@app.post("/classify")
def classify(note: NoteIn, request: Request):
    return request.app.state.classifier.predict(note.text)


@app.post("/notes", status_code=201)
def create_note(note: NoteIn, request: Request):
    result = request.app.state.classifier.predict(note.text)
    return db.add_note(note.text, result)


@app.get("/notes")
def get_notes(category: str | None = None):
    if category and category not in LABELS:
        raise HTTPException(400, f"category must be one of {sorted(LABELS)}")
    return db.list_notes(category)


@app.get("/search")
def search(q: str = ""):
    results = search_tfidf(q, db.list_notes())
    return {"query": q, "results": results,
            "message": "" if results else "No notes match well."}


@app.post("/starter/load")
def starter_load(request: Request):
    return db.load_starter(request.app.state.classifier)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
