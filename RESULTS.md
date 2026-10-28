# RESULTS

All numbers below come from running the commands on my own machine (Windows, Python 3.11, scikit-learn 1.9.1, numpy 2.4.6, sentence-transformers 6.2.0, random seed 42). Validation labels were never used for fitting.

## Dataset and training
- Training rows: 156
- Validation rows: 50
- Command that reproduces these numbers: `python train.py` (and `python cv_compare.py` for the cross-validation table)
- Baselines: majority class ("Study"), and hand-written keyword rules (word lists written from `CATEGORY_GUIDE.md` before looking at any results; never used inside the app).
- Trained model: scikit-learn pipeline: TF-IDF on words (1-2 grams, English stop words removed) joined with TF-IDF on character 3-5 grams (`char_wb`), then `LogisticRegression(C=10, max_iter=1000, random_state=42)`. Saved to `model/model.joblib`.
- Library versions and seed: scikit-learn 1.9.1, numpy 2.4.6, seed 42. Running `train.py` twice gives identical output.

## Metrics (validation dataset only)

| Approach | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Baseline: majority class (Study) | 0.300 | 0.115 |
| Baseline: keyword rules | 0.720 | 0.733 |
| Trained v1: word TF-IDF + LogReg (first model) | 0.640 | 0.666 |
| **Trained final: word + character TF-IDF + LogReg** | **0.880** | **0.876** |

Per-category results, final model (validation):

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Study | 0.833 | 1.000 | 0.909 | 15 |
| Money | 1.000 | 1.000 | 1.000 | 11 |
| Travel | 0.889 | 0.615 | 0.727 | 13 |
| Shopping | 0.833 | 0.909 | 0.870 | 11 |

Confusion matrix, final model (rows = true, columns = predicted):

| | Study | Money | Travel | Shopping |
| --- | ---: | ---: | ---: | ---: |
| Study | 15 | 0 | 0 | 0 |
| Money | 0 | 11 | 0 | 0 |
| Travel | 3 | 0 | 8 | 2 |
| Shopping | 0 | 0 | 1 | 10 |

The first trained model scored *below* the keyword rules. Its main error was sending 8 Study and 6 Money validation notes to Travel; the validation notes use words that never appear in the training data (for example railway, quiz, coursework, credit, internet).

## Error analysis (final model: 6 mistakes, 3 distinct)

Notes that differ only in a day, time or place count as one mistake.

| Note | Expected | Predicted | Possible reason |
| --- | --- | --- | --- |
| "I must reach the railway station by 10 AM / 2 PM / 8 AM." (3 notes) | Travel | Study | "railway" never appears in training. The number "10" and word pieces of "station" (such as "ati", "st ") also occur in Study notes ("statistics", "10 AM tutorial"). |
| "Our return journey from Jaipur is on Wednesday / Friday." (2 notes) | Travel | Shopping | "return" appears in 4 training notes, all Shopping ("Return the damaged backpack to the store"); "journey" never appears in training. |
| "Put snacks and cleaning items in the cart." | Shopping | Travel | Almost no known vocabulary: "cart" and most words are absent from training, and the best similarity to any training note is only 0.11. |

**Evidence (mistake 1, "I must reach the railway station by 10 AM."):**
- Predicted probabilities: Study 0.56, Travel 0.25, Money 0.10, Shopping 0.08.
- Three most similar training notes: "Find directions from the station to our hotel." (Travel, similarity 0.32), "The boarding time for the plane is 10 AM." (Travel, 0.24), "Attend the chemistry tutorial at 10 AM." (Study, 0.21).
- Counts in `train.csv`: "railway" 0 notes; "station" 1 note (Travel); "10 am" 4 notes (3 Study, 1 Travel).
- Feature contributions (model coefficients x TF-IDF value, Study minus Travel): the word "10" pushes toward Study (+0.21); the word "station" pushes strongly toward Travel (-0.57) but is outweighed by several character pieces that belong to Study notes. So the model had a correct signal and lost to misleading ones.

**Evidence (mistake 2, "return journey"):** the word "return" alone contributes +0.52 toward Shopping versus Travel, while "Jaipur" contributes -0.23 toward Travel. The word-sense of "return" (to give back a purchase versus to travel back) cannot be learned from 4 training notes that only use the first sense. Note that this mistake is a *regression*: the first model got both "return journey" notes right and the final model gets them wrong.

## Search evaluation

- Query file and command: `data/search_queries.json`, `python eval_search.py`
- Number of queries: 16 over the 12 starter notes; 8 of them share no content words with their target note (money transfer, accommodation, groceries, airplane trip, send back faulty item, housing payment, upcoming exam, when do I need to submit coursework?).

| Search approach | Hit rate @3 | Hard (no shared word) queries solved |
| --- | ---: | ---: |
| Keyword overlap baseline | 0.50 (8/16) | 0/8 |
| Word TF-IDF cosine | 0.50 (8/16) | 0/8 |
| Character TF-IDF cosine (stop words removed) | 0.69 (11/16) | 3/8 |
| Sentence embeddings (all-MiniLM-L6-v2) | **1.00 (16/16)** | **8/8** |

- **Approach used in the app:** sentence embeddings, because it is the only method that finds notes sharing no words with the query. If the model cannot be loaded, the app falls back to character TF-IDF.
- **Low-relevance cutoff:** results with a similarity below 0.27 are hidden. Chosen between the highest score of any of 3 unrelated queries (0.25) and the lowest score of a correct top result (0.29). A cutoff of 0.30 hid two real queries. The margin is only 0.02 on each side and rests on 3 unrelated queries, so it is a judgement, not a precise optimum.
- **A query that still fails:** "something about money" (written as an extra probe after seeing the first results, so it is not part of the 16): a person would expect N002, N006 or N010, but the embedding search returns no result above the cutoff, because a vague category-level phrase is not close to any single note. Lowering the cutoff would find it but let unrelated queries through.
- Other extra probes (not counted in the hit rate): "remember to call mom" returns nothing (correct); "computer accessories" finds N004; "cheap flights" finds N003; "how do I get to the airport?" ranks the railway-station note N011 above the flight note N003.
- Measurement bug found and fixed: the first version gave keyword and word TF-IDF 0.56 because notes with score 0 were ranked by ID and counted as hits. After requiring score > 0 they are 0.50.

## Notes that fit no category

- **How `uncertain` is decided:** from the model's own output: a note is uncertain when its highest predicted class probability is below a threshold (0.55) stored in `model/config.json`. No keyword list.
- **How the threshold was chosen:** `evaluate_uncertainty.py`. The 12 out-of-scope notes were split 6 / 6 with seed 42: a tuning half and a held-out half. A rule fixed in advance chose the threshold from the tuning half and validation only: maximise (tuning-half out-of-scope flagged) minus (validation flagged), rejecting any threshold that flags more than one third of validation notes. The held-out half is only reported.
- **Did I train on `out_of_scope_examples.csv`?** Not for the chosen method. For the alternative method (an extra "Other" class), the tuning half (6 notes) was used for training and the held-out half (6 notes) was kept aside for evaluation.

| Setting (max probability below...) | Out-of-scope flagged: held-out 6 | Out-of-scope flagged: tuning 6 | Validation notes flagged |
| --- | ---: | ---: | ---: |
| **Chosen: 0.55** | **100% (6/6)** | 83% (5/6) | **30% (15/50)** |
| Alternative: 0.45 | 100% (6/6) | 50% (3/6) | 18% (9/50) |
| Alternative: 0.35 | 50% (3/6) | 33% (2/6) | 8% (4/50) |
| Other method: extra "Other" class, 0.55 | 100% (6/6) | (trained on, not meaningful) | 32% |

- **Why this trade-off:** 0.55 flags all held-out out-of-scope notes while staying under the one-third limit on validation notes. A lower threshold misses out-of-scope notes; a higher one flags a third or more of normal notes. At 0.55 the flag also catches 5 of the 6 real classifier mistakes; 10 of the 44 correctly classified validation notes are flagged too (a 23% false-alarm cost). The extra "Other" class method did not beat the simple threshold on the held-out half, so the simple one was kept.

## What I tried to fix, and whether it worked
- **Change:** replaced word-only TF-IDF with word TF-IDF without English stop words joined with character 3-5 gram TF-IDF.
- **How it was chosen (without fitting on validation labels):** `cv_compare.py` runs 5-fold GroupKFold on `train.csv` only. The training notes were grouped into 56 templates so that near-duplicates never sit in different folds. Grouped-CV macro-F1: current model 0.647; lower C=1 0.558; class weighting 0.666; stop words removed 0.765; character n-grams 0.792; **stop words + character n-grams 0.854** (best, chosen).
- **Result on validation, looked at once after the choice:** accuracy 0.640 to 0.880, macro-F1 0.666 to 0.876. It fixed 14 validation mistakes (internet bill, credit card, quiz, coursework, "study for next week") and introduced 2 new ones ("return journey"). Honesty note: before choosing the fix, an exploratory look at the first model's confidence values showed that no uncertainty threshold could work on it; no model setting was chosen from that look.

## What I would improve next
- Collect more varied training notes (more vocabulary, words with two senses such as "return").
- Use a pretrained sentence-embedding model as classifier features and compare it with grouped cross-validation.
- Calibrate the probabilities, and choose the uncertainty threshold with more out-of-scope examples.
- Build a larger search test set, written by someone other than the developer.

## Limitations
- The dataset is small and synthetic, and cannot demonstrate real-world reliability on all kinds of notes.
- **Trust in the validation score:** moderate to low. The 50 validation notes come from a handful of templates, so the 6 mistakes are only 3 independent mistakes; one more template wrong would move accuracy by 4-6 points. The validation set also uses vocabulary absent from training, so it is a harder, shifted test, which is useful, but it means 0.880 should not be read as a general accuracy.
- The search test set (16 queries) was written by the developer who also built the search, so 1.00 is optimistic.
- The uncertainty result rests on 12 out-of-scope notes (6 held out); "100% flagged" means 6 of 6, not a general guarantee, and about 30% of validation notes are flagged.
