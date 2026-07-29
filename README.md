# Yoruba Fake News Detection

Classifies Yoruba-language news text as **GENUINE** or **FAKE**. MIT dissertation project (Fatunwase Micheal).

The pipeline: scrape genuine news + fact-checked fake claims → validate/merge the corpus → build preprocessing variants → train classical and transformer models → evaluate → serve predictions via a Streamlit app and a FastAPI endpoint.

## 1. Prerequisites

- Python 3.10+
- Git
- ~2 GB free disk space (more if you train transformer models, which download multi-hundred-MB checkpoints)

## 2. Install on a new computer

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

## 3. Running the pipeline

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

## 4. Running the app

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

## 5. Project structure

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
```

## 6. Known limitations (current state)

- Only 2 scraper sources are verified working from this environment: **BBC News Yoruba** (genuine) and **Dubawa's Yoruba fact-check category** (fake). Voice of Nigeria and Alaroye returned unreachable/404 when last checked — their selectors in `config/sources.yaml` are unverified guesses; re-check against a live page before relying on them.
- `config/config.yaml` targets a 3,000–4,000 row balanced corpus from multiple sources per class. With only one source per class, `src.annotation.validate_corpus`'s leakage check will (correctly) flag that a model can trivially learn "source style" instead of "genuine vs. fake content" — don't trust reported accuracy until there are multiple independent sources per class.
- Transformer training (AfriBERTa, AfroXLM-R) is implemented but untested end-to-end in this environment (no GPU, and `sentencepiece` was only added to `requirements.txt` after the fact — reinstall requirements if you hit a tokenizer error).