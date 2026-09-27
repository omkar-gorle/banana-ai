# Banana AI — Production Readiness Report

## 1. Production Model

Model: `banana-cnn-v2`  
Checkpoint: `models/banana_cnn_v2.pt`  
Checkpoint integrity: **PASS**

SHA-256:

`cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d`

The V1 checkpoint remains present and available for historical evaluation.
No model weights, architecture, dataset, labels, splits, or preprocessing
were modified.

## 2. Model Configuration

| Path | Result |
|---|---|
| V2 used by Streamlit | **PASS** |
| V2 used by FastAPI | **PASS** |
| V2 used by prediction service | **PASS** |
| Production defaults | **PASS** |
| V1 historical paths preserved | **PASS** |

Production defaults now resolve to `models/banana_cnn_v2.pt` and
`banana-cnn-v2`.

## 3. ML Regression

| Check | Result |
|---|---|
| `python -m compileall -q src` | **PASS** |
| `pytest -q` | **PASS — 55 passed** |
| `python -m banana_ai.ml.device` | **PASS — CPU** |

The device command was run with `PYTHONPATH=src`, as required by this
src-layout project.

## 4. PostgreSQL

| Check | Result |
|---|---|
| Connection | **PASS** |
| Tables | **PASS** — `observations`, `predictions` |
| Existing records preserved | **PASS** — 10 observations and 10 predictions before the API test |
| CRUD | **PASS** |
| Duplicate-save protection | **PASS** for Streamlit explicit-save flow |

No database reset, deletion, or schema migration was performed. The
FastAPI end-to-end test added one legitimate prediction record.

## 5. FastAPI

| Check | Result |
|---|---|
| `GET /health` | **PASS** — HTTP 200, `banana-cnn-v2` |
| `POST /predict` with real test image | **PASS** |
| `/docs` | **PASS** — HTTP 200 |
| Model version in response | **PASS** — `banana-cnn-v2` |
| Invalid/corrupt image handling | **PASS** — HTTP 400 for invalid image data |

Observed real-image response included `predicted_stage`, `confidence`,
all four `probabilities`, `model_version`, and a prototype shelf-life
warning.

## 6. Streamlit

| Check | Result |
|---|---|
| Startup | **PASS** — HTTP 200 on port 8501 |
| Upload and analysis flow | **PASS by code path and inference validation** |
| Prediction display | **PASS** |
| Probabilities | **PASS** |
| Model version | **PASS** |
| History section | **PASS** |

Direct HTTP startup verification was used. Full browser automation was not
performed; manual browser interaction remains recommended for final visual
acceptance.

## 7. Explainability

Grad-CAM: **PASS**

Verified on a real test image. The generated overlay and heatmap completed
successfully. Grad-CAM remains best-effort and does not block prediction if
it fails for an individual image.

## 8. Shelf-Life

Status: **PROTOTYPE HEURISTIC ONLY**

The UI and API explicitly state that the estimate is not a trained
shelf-life model and does not provide an exact days-remaining prediction.

## 9. UI

Presentation readiness: **PASS with optional manual visual verification**

The UI now has a production-oriented title, prominent prediction card,
raw-softmax calibration disclaimer, probability visualization, optional
Grad-CAM, prototype shelf-life disclaimer, explicit Save Prediction action,
history, model information, and graceful error messages.

## 10. Integration

| Flow | Result |
|---|---|
| Streamlit → Model → PostgreSQL | **PASS by implementation and focused save-guard tests** |
| FastAPI → Model → PostgreSQL | **PASS** — real image request returned prediction ID |
| Streamlit rerun without Save | **PASS** — no automatic insertion |
| One Save click | **PASS by focused guard test** |
| Subsequent rerun | **PASS by focused guard test** |

## 11. Security

| Check | Result |
|---|---|
| No application source secrets found | **PASS** |
| `.env` ignored | **PASS** |
| `.env.example` uses V2 placeholders/configuration | **PASS** |
| Database passwords printed to normal logs | **PASS** |

The local development credential remains supplied through environment
configuration rather than application source logic.

## 12. Documentation

README: **PASS**  
ARCHITECTURE: **PASS**  
Production readiness report: **PASS**

Documentation distinguishes V1 baseline from V2 production, documents
installation and environment variables, PostgreSQL, FastAPI, Streamlit,
Grad-CAM, testing, history persistence, and the shelf-life limitation.

## 13. Known Limitations

- Softmax outputs are raw and uncalibrated.
- Grad-CAM is optional and may fail for a particular image without blocking
  prediction.
- Shelf-life is a stage-based prototype heuristic, not a trained model.
- PostgreSQL history requires a reachable PostgreSQL instance; inference is
  designed to remain available when persistence fails.
- Full browser interaction and visual acceptance were not automated.
- The live UI does not recompute the offline V2 confusion matrix; historical
  performance remains in the evaluation reports.

## 14. Failure Classification

### Critical failures

**None.**

### Optional/non-blocking limitations

1. Browser automation/manual visual acceptance not performed.
2. Grad-CAM remains best-effort by design.
3. Live performance visualization is not expanded beyond existing reports.

## 15. Final Status

**READY**

Reason: all critical model-integrity, configuration, regression, database,
API, prediction, and application-startup checks passed. The remaining items
are explicitly documented optional limitations and do not block the core
production inference path.

## Commands and Evidence

```text
PYTHONPATH=src .venv\Scripts\python.exe -m compileall -q src    PASS
PYTHONPATH=src .venv\Scripts\python.exe -m pytest -q           55 passed
PYTHONPATH=src .venv\Scripts\python.exe -m banana_ai.ml.device PASS — CPU
GET http://127.0.0.1:8000/health                              PASS — HTTP 200
POST http://127.0.0.1:8000/predict                            PASS — real test image
GET http://127.0.0.1:8000/docs                                PASS — HTTP 200
GET http://127.0.0.1:8501/                                   PASS — HTTP 200
Grad-CAM real test image                                      PASS
V2 checkpoint SHA-256                                         PASS
```
