# AI_WORKFLOW

## 1. Tools
- **Claude** (chat assistant) for planning the phases, writing and explaining code, and producing explainer documents. It ran the code in its own workspace first; I re-ran everything on my machine and only my output is quoted in `RESULTS.md`.
- VS Code and PowerShell, Python 3.11 with a virtual environment, scikit-learn, FastAPI, SQLite, sentence-transformers (all-MiniLM-L6-v2), pytest, Git and GitHub.
- No paid APIs and no hosted LLM inside the app.

## 2. One delegated task (prompt as I wrote it)
> "hii i need to complete a technical assessment for the internship opportunity of ai engineer i need you to give me a complete phase by phase guide where to implement , when to push to github and complete implementation task . Refer to the zip file troughly and provide the best and fesible solution and platforms to use and prerequisites to be downloaded provide me the phase by phase implementation stack then we will implement it together"

**What I kept:** the phase-by-phase plan (setup, baseline and model, uncertainty, API, notes app, search evaluation, tests and documents), one FastAPI process serving the API and a plain HTML page, SQLite for storage, scikit-learn TF-IDF + logistic regression, and a push to GitHub after each phase.
**What I changed or asked for:** I asked for a PDF explaining each phase so I understand it conceptually, and a master hand-over PDF. After the first model scored below the keyword baseline, the order of phases changed: the cross-validated model fix and error analysis were moved before the uncertainty threshold, because probabilities from the weak model could not separate out-of-scope notes. I also skipped the optional bonus tasks and shortened the plan to fit the deadline.

## 3. Problems caught
1. **Search evaluation flattered the baselines.** When a query matched nothing, every note scored 0 and was ranked by ID, so "upcoming exam" counted as a hit for N001 by luck. Symptom: keyword and TF-IDF scored 0.56 although neither can match different words. Fix: a note with score 0 is never a result. Proof: before/after, 9/16 became 8/16 (0.56 to 0.50), re-run with `python eval_search.py`.
2. **Character n-gram search matched filler words.** "what do I need to buy?" returned the Pune train note. Found by trying the query in the browser; fixed by removing stop words before building character n-grams; checked by the same query returning the shopping notes and by the unchanged hit rate.
3. **The app was still using the old search settings** after an update: the page showed a character-TF-IDF-style score for "money transfer". Found by reading the relevance score and the evaluation output (cutoff still 0.30); fixed by patching `app/search.py`; checked by `search('money transfer')` returning N006 from the embedding method.
4. **Trained model lost to the keyword rules** (0.640 vs 0.720). Found in the first `train.py` run; explained with feature and nearest-note evidence in `RESULTS.md`; fixed with a fix chosen by grouped cross-validation, not by watching validation.
5. **Environment issues:** a numpy version pinned for Python 3.12 failed on my Python 3.11 (fixed by pinning 2.4.6); `git add .` staged the whole `.venv` because `.gitignore` was missing (fixed by recreating it and checking `git ls-files`); `pytest` could not import `app` (fixed with a `pytest.ini` that sets `pythonpath = .`).
6. **A too-optimistic claim about the search cutoff.** The assistant said a 0.27 cutoff "keeps all 16" real queries. My clean-clone run printed 15/16 answered, so one query was hidden. I fixed the evaluation to also report the hit rate with the cutoff applied (0.94, and the hidden query is named) and corrected `RESULTS.md`. Checked by re-running `python eval_search.py`.

Approximate time spent: [FILL IN, hours].
