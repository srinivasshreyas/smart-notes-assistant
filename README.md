# Smart Notes Assistant

A small local app that saves notes, classifies each one into **Study / Money / Travel / Shopping** with a classifier trained in this repository, says when it is **unsure**, and lets you **search** your notes by meaning.

Everything runs locally: no paid service, no API key, no GPU. All data in the repository is synthetic.

## Requirements
- Python 3.11 (developed and tested on Windows with Python 3.11; should work on 3.11+).
- About 1 GB of disk space (PyTorch is installed as part of `sentence-transformers`).
- **Internet on the first search only:** the semantic search downloads the `all-MiniLM-L6-v2` model (about 90 MB) from Hugging Face the first time it is used. If the model cannot be downloaded or loaded, the app **automatically falls back to character TF-IDF search** (lower quality, no crash). Classification never needs the internet.

## Install (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Train the classifier (one command)
```powershell
python train.py
```
Reads `data/train.csv`, evaluates two baselines and the trained model on `data/validation.csv`, prints accuracy / macro-F1 and a confusion matrix, and saves `model/model.joblib`. A trained model is already committed, so you can skip this step and start the app directly.

The uncertainty threshold lives in `model/config.json` and is produced by:
```powershell
python evaluate_uncertainty.py
```
(Re-run it after any change to the model, then commit the new `config.json`.)

## Start the app
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- App (add, list, filter, search): http://127.0.0.1:8000
- Classification API: `POST http://127.0.0.1:8000/classify`

## Load the starter notes (original IDs N001-N012)
Either click **Load starter notes** in the page, or run:
```powershell
python -m app.db load-starter
```
Loading twice never creates duplicates. Notes you add get IDs from `N013` upwards; `N001`-`N012` are never reused. Notes are stored in `data/notes.db` (SQLite) and survive restarts.

## Use /classify
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify -ContentType "application/json" -Body '{"text":"Pay the electricity bill tomorrow."}'
```
Returns `{"category": "...", "uncertain": true|false, "confidence": 0.xx}`. Invalid input (empty, whitespace, missing field, null, non-string, malformed JSON, over 1,000 characters) returns HTTP 422 with a readable message, and the server keeps running.

Reviewer smoke test:
```powershell
python examples\test_api.py --url http://127.0.0.1:8000/classify
```

## Tests
```powershell
pytest
```
Runs 16 tests (classification, invalid input, determinism, a 1,000-character note with the rupee sign, persistence, starter loading without duplicates, filtering, search). Tests use a temporary database and never touch `data/notes.db`.

## Evaluate search
```powershell
python eval_search.py
```
Prints hit rate at 3 for each search method, failing queries, the low-relevance cutoff check and extra probe queries. Query set: `data/search_queries.json`.

## How it works
- **Classifier:** scikit-learn pipeline: word TF-IDF (stop words removed, 1-2 grams) joined with character 3-5 gram TF-IDF, then logistic regression (C=10, seed 42).
- **Uncertain flag:** a note is `uncertain` when the highest class probability is below the threshold in `model/config.json` (0.55). The category shown is still the best guess among the four labels, and uncertain notes are marked with a badge in the page. Filtering by a category includes uncertain notes whose best guess is that category.
- **Search:** sentence embeddings (`all-MiniLM-L6-v2`) with cosine similarity and a minimum-relevance cutoff (0.27); results below it are hidden, and the page says "No notes match well."

See `RESULTS.md` for the measurements and `AI_WORKFLOW.md` for how this was built.

## Project layout
```
app/            FastAPI app: main.py (routes), classifier.py, db.py, search.py, static/index.html
data/           supplied data + search_queries.json
model/          model.joblib, config.json
tests/          pytest tests
train.py, cv_compare.py, analyze_errors.py, evaluate_uncertainty.py, eval_search.py
```

## Limitations
- The data is small, synthetic and templated: scores show the method works on this data, not how it would perform on real notes.
- About 30% of validation notes are flagged uncertain at the chosen threshold, so the app is cautious.
- The first embedding search after starting the server takes a few seconds while the model loads.
- Local use only: no authentication, one user, SQLite file.
