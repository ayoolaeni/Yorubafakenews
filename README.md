# Yoruba Fake News Detection

Classifies Yoruba-language news text as **GENUINE** or **FAKE**. MIT dissertation project (Fatunwase Micheal).

The pipeline: scrape genuine news + fact-checked fake claims → validate/merge the corpus → build preprocessing variants → train classical and transformer models → evaluate → serve predictions via a Streamlit app and a FastAPI endpoint.

## 1. How this project works (architecture)

**This is a classical machine-learning text classifier, not a RAG system and not an LLM.** At prediction time it does not retrieve documents and does not call any language model (local or hosted) — a request goes straight into a scikit-learn pipeline and comes back with a label. The stages, in order:

1. **Scraping** (`src/scraping/`) — pulls genuine articles and fact-checked "fake" claims from Yoruba-language sources listed in `config/sources.yaml` (currently BBC News Yoruba for genuine, Dubawa's Yoruba fact-check category for fake) into `data/raw/`.
2. **Annotation / validation** (`src/annotation/validate_corpus.py`) — merges the raw scrapes into one labelled corpus, deduplicates near-identical rows, filters by length, and flags source-leakage risk (a model scoring high just by learning "which outlet wrote this" rather than "is this true").
3. **Preprocessing** (`src/preprocessing/pipeline.py`) — builds four text variants per article (diacritics kept/stripped × stopwords kept/removed, using `fixtures/stopwords_yoruba.txt`), counts emoji as a numeric feature, and writes one frozen 70/15/15 train/val/test split (`data/splits/`) that every model is trained and scored against.
4. **Feature extraction + model training** (`src/models/`) — each text variant is turned into TF-IDF / bag-of-words features and fed to four **classical scikit-learn classifiers** defined in `config/models.yaml`: Naive Bayes, Linear SVM, Logistic Regression, Random Forest, each grid-searched with cross-validation. A **transformer fine-tuning path** also exists (`src/models/train_transformer.py`) for two pretrained African-language encoders — `castorini/afriberta_large` and `Davlan/afro-xlmr-base` — fine-tuned with a classification head on the same split. This path is implemented but not what's currently deployed (see Known Limitations).
5. **Evaluation** (`src/evaluation/report.py`) — every trained model is scored once against the held-out test split; results are ranked by macro F1 into `results/metrics/leaderboard.csv`, and the top row is written to `results/metrics/best_model.json`. `src/evaluation/error_analysis.py` then isolates what the winning model gets wrong.
6. **Serving** (`app/`) — `app/streamlit_app.py` and `app/api.py` both call the same `src.models.predictor.load_best_predictor()`, which reads `best_model.json`, loads that exact model file, and preprocesses incoming text with the same variant logic used in training — so what a user types goes through the identical pipeline the model was trained on.

The model currently selected as best (see `results/metrics/best_model.json`) is a **Logistic Regression** classifier on the `diacritic_preserved__stopwords_kept` TF-IDF variant — no embeddings API, no vector database, no prompt engineering involved.

## 2. Prerequisites

- Python 3.10+
- Git
- ~2 GB free disk space (more if you train transformer models, which download multi-hundred-MB checkpoints)

## 3. Install on a new computer (without Docker)

Clone the repo, then from the project root:

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

`pip install -e .` installs this project itself (via `setup.py`) in editable mode, so `src` is importable no matter how a script is invoked (`pytest`, `streamlit run`, `uvicorn`, plain `python`). Without it you'll see `ModuleNotFoundError: No module named 'src'`.

Then open `.env` and set `RESEARCH_CONTACT_EMAIL` — it's embedded in the scraper's User-Agent string so sites can identify/contact the project (see `config/sources.yaml`).

### Verify the install

```bash
pytest -v
```

You should see all tests pass (config loading, scraping parsers, corpus validation, preprocessing, model training, inference — none of these need real scraped data or a GPU).

## 4. Running the pipeline

Each stage reads config from `config/config.yaml`, `config/sources.yaml`, and `config/models.yaml`. Run them in order (or use the matching `make` target):

| Stage | Command | Makefile target |
|---|---|---|
| Scrape genuine articles | `python -m src.scraping.run_scrapers --source all --class genuine --limit 500` | — |
| Scrape fake/fact-checked claims | `python -m src.scraping.run_scrapers --source all --class fake` | `make corpus` (runs both + validation) |
| Validate/merge corpus | `python -m src.annotation.validate_corpus` | (included in `make corpus`) |
| Build preprocessing variants | `python -m src.preprocessing.pipeline --build-all-variants` | `make preprocess` |
| Build the train/val/test split | `python -m src.preprocessing.pipeline --make-splits` | (included in `make preprocess`) |
| Train models | `python -m src.models.train --model all --features all` | `make train` |
| Evaluate + rank models | `python -m src.evaluation.report` | `make evaluate` |
| Error analysis on the best model | `python -m src.evaluation.error_analysis --model best` | (included in `make evaluate`) |

`--source` accepts a specific source name (e.g. `"BBC News Yoruba"`) or `all`. `--model` accepts a specific model name from `config/models.yaml` (e.g. `svm`, `naive_bayes`, `afriberta`) or `all`. Scraping is resumable — rerunning a stage skips URLs/records already collected.

**Note on `--model all`:** this also attempts the transformer checkpoints (AfriBERTa, AfroXLM-R) listed in `config/models.yaml`. Those need `torch`/`transformers`/`datasets`/`accelerate`/`sentencepiece` (all in `requirements.txt`) plus internet access to Hugging Face Hub, and are slow without a GPU. A checkpoint that fails to load or train is skipped with a warning — it won't stop classical models from training on the other variants. To skip transformers entirely, pass a specific classical model name (or run it once per name: `naive_bayes`, `svm`, `logistic_regression`, `random_forest`).

**The split is written once.** `--make-splits` refuses to overwrite an existing `data/splits/*.csv` unless you pass `--force` — this is deliberate, so the test set can't accidentally leak into training after modelling has started.

## 5. Running the app (without Docker)

Once `make train` and `make evaluate` have produced `results/metrics/best_model.json`:

```bash
streamlit run app/streamlit_app.py     # interactive UI at http://localhost:8501
```
or
```bash
uvicorn app.api:app --reload --port 8000   # POST /predict at http://localhost:8000
```

Example API call:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "some Yoruba news text here"}'
```
Response: `{"label": "genuine"|"fake", "confidence": 0.0-1.0, "model": "...", "variant": "..."}`

Before any training has run, both the app and API detect the missing model and respond accordingly (a friendly message in Streamlit; HTTP 503 from the API) instead of crashing.

## 6. Running with Docker (recommended — no Python setup needed)

This repo ships with a `Dockerfile` and `docker-compose.yml` that serve the **already-trained model** (the `models/` and `results/` folders are included in this project copy, so no training/scraping is required). Requires only [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

The image installs `requirements-app.txt`, a lean subset of `requirements.txt` covering only what `app/streamlit_app.py` and `app/api.py` actually import at runtime (pandas, scikit-learn, streamlit/fastapi stack) — it skips scraping, transformer fine-tuning, and plotting dependencies, which the serving path never touches. This keeps the image small and the build fast. If you need to scrape/train/evaluate *inside* a container too, install from `requirements.txt` instead.

```bash
docker compose up --build app     # Streamlit UI at http://localhost:8501
```
or
```bash
docker compose up --build api     # FastAPI at http://localhost:8000
```
or both at once:
```bash
docker compose up --build
```

Stop with `Ctrl+C`, or `docker compose down` to remove the containers. See `HOW_TO_RUN_WITH_DOCKER.txt` for a non-technical, click-by-click version of these steps.

## 7. Project structure

```
config/            project/split/corpus/preprocessing settings, scraper targets, model grids
src/scraping/       fetches + parses genuine/fake sources into data/raw/
src/annotation/     merges + validates the raw corpus (dedup, length filter, source-leakage check)
src/preprocessing/  diacritic/stopword variants, emoji-as-feature, the frozen train/val/test split
src/models/         classical (sklearn) + transformer (HF) training, inference, single-text prediction
src/evaluation/     test-set metrics, leaderboard, confusion matrices, error analysis
app/                Streamlit UI and FastAPI service, both backed by src.models.predictor
tests/              pytest suite — all synthetic data, no network/training required
data/, models/, results/   generated artefacts (gitignored except manifests/placeholders)
Dockerfile, docker-compose.yml, .dockerignore, requirements-app.txt   container build for app/api (see section 6)
```

## 8. Known limitations (current state)

- Only 2 scraper sources are verified working from this environment: **BBC News Yoruba** (genuine) and **Dubawa's Yoruba fact-check category** (fake). Voice of Nigeria and Alaroye returned unreachable/404 when last checked — their selectors in `config/sources.yaml` are unverified guesses; re-check against a live page before relying on them.
- `config/config.yaml` targets a 3,000–4,000 row balanced corpus from multiple sources per class. With only one source per class, `src.annotation.validate_corpus`'s leakage check will (correctly) flag that a model can trivially learn "source style" instead of "genuine vs. fake content" — don't trust reported accuracy until there are multiple independent sources per class.
- Transformer training (AfriBERTa, AfroXLM-R) is implemented but untested end-to-end in this environment (no GPU, and `sentencepiece` was only added to `requirements.txt` after the fact — reinstall requirements if you hit a tokenizer error).