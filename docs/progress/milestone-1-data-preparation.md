# 📊 Milestone 1 Progress Report — Data Preparation & Management

**Status:** ✅ **COMPLETE**
**Project:** Structured Data Extraction Using Small Language Models
**Report date:** 2026-10-21

---

## 1. Objective

Establish deterministic, immutable, schema-conformant data artifacts that every
downstream milestone (model selection, RAG, Text-to-SQL, evaluation, deployment)
depends on.

---

## 2. Datasets

Two engineered datasets are produced from a single upstream source, with a
deterministic synthetic fallback for offline environments.

| Dataset | Type | Task | Rows | Source |
| --- | --- | --- | --- | --- |
| `sephora_pricing` | Regression | Predict `price_usd` | 6,344 | Sephora catalogue |
| `skincare_compatibility` | Classification | Predict `compatibility_tier` | 7,396 | Sephora skincare + dermatological scoring |

**Upstream source:** Hugging Face Hub dataset `MayaKitzis/sephora_products`
(`product_info.csv`), staged locally at `data/raw/product_info.csv`. Provenance in
the artifacts confirms the **primary (real) dataset** was used — 285 distinct real
brands (e.g., The Ordinary, CeraVe, La Roche-Posay) with real Sephora product IDs
(`P473668`, …).

**Synthetic fallback:** `generate_synthetic(seed=42)` reproduces both schemas
deterministically when offline (`SLM_MLOPS_OFFLINE=1`) or when the source is
unreachable — guaranteeing reproducible CI runs with zero network dependency.

---

## 3. Pipeline Stages Implemented

The `main.py` entry point executes a fully deterministic pipeline:

1. **Tiered ingestion** (`parsers/source.py`) — primary Sephora CSV/Hub mirror → synthetic fallback
2. **Deterministic feature engineering** — INCI ingredient parsing, active-ingredient detection, comedogenic scoring, fragrance/alcohol flags, size normalization (oz→ml), concentration ranking
3. **Exact deduplication** (`processors/cleaner.py`)
4. **Leak-free stratified split** (`processors/splitter.py`) — 70/15/15 grouped by `product_id` with cross-contamination verification
5. **Min-max scaler fitting** (`processors/scaler.py`) — fitted strictly on the training split only
6. **Artifact serialization** — Parquet + validated JSON, all written read-only

---

## 4. Artifacts Produced (`src/artifacts/`)

| Artifact | Contents | Verified |
| --- | --- | --- |
| `sephora_pricing_normalized.parquet` | 6,344 rows, 12 columns | ✅ |
| `pricing_train/val/test.parquet` | 4,440 / 952 / 952 (70/15/15) | ✅ |
| `skincare_compatibility_clean.parquet` | 7,396 rows, 10 columns | ✅ |
| `skincare_train/val/test.parquet` | 5,172 / 1,112 / 1,112 (70/15/15) | ✅ |
| `regression_scaler.json` | Min/max bounds for 7 numeric features | ✅ |
| `rag_chunks.json` | 27 schema-metadata RAG chunks | ✅ |
| `faiss_store/index.faiss` + `index.pkl` | 384-dim schema vector index | ✅ |

### Key distribution facts

- **Pricing categories:** Skincare 1,906 · Makeup 1,648 · Fragrance 1,255 · Hair 1,175 · Bath & Body 360
- **Skincare compatibility tiers:** `Irritation_Risk` 3,426 · `Neutral` 2,035 · `Holy_Grail_Safe` 1,935
- **Scaler bounds (price_usd):** $3.00 – $395.00

---

## 5. Schema Contracts (Deterministic Boundaries)

Defined centrally in `schemas.py` and enforced via Pydantic v2:

- **`sephora_pricing`** (12 cols): `product_id`, `brand_name`, `product_category`, `size_ml`, `active_ingredient_count`, `concentration_rank`, `loves_count`, `reviews_count`, `average_rating`, `sephora_exclusive`, `clean_at_sephora`, `price_usd`
- **`skincare_compatibility`** (10 cols): `product_id`, `brand_name`, `product_category`, `primary_active`, `has_fragrance`, `has_alcohol_denat`, `comedogenic_rating`, `skin_type`, `skin_tone`, `compatibility_tier`

---

## 6. RAG Chunk Construction (Bridging to Milestone 3)

`build_rag_chunks()` pre-generates the schema-metadata chunks that the FAISS schema
index consumes:

- 2 dataset-overview chunks + 22 column chunks + 3 domain-alias chunks = **27 total**
- Each chunk carries natural-language aliases (e.g., `price_usd` → "cost", "how much", "affordable") for later semantic retrieval
- FAISS index built with a deterministic 384-dim `HashingVectorizer` embedder (runtime swaps to `nomic-embed-text` in a later milestone)

---

## 7. Test Coverage & Verification

| Test module | Focus | Tests |
| --- | --- | --- |
| `tests/test_parsers.py` | Primary parser mapping against deterministic fixture | 2 |
| `tests/test_data_preprocessing.py` | Ingestion, dedup, split, contamination, scaler | 14 |
| `tests/test_rag.py` | Embedding determinism, scaler roundtrip, FAISS persistence | 3 |

**Verified invariants:** offline determinism, 70/15/15 proportions, zero
cross-split product leakage, strict scaler bounds, schema conformance, and lossless
JSON roundtrip. All tests run offline with no live Ollama dependency.

---

## 8. Quality Gate Status

The full gate (`bash scripts/check.sh`) was run and passes for this milestone. Note:
the *evaluation threshold* gates (`test_eval_thresholds.py`, ≥85% SQL success,
≤10% MAPE) belong to Milestone 5 and are not yet active — the
`tests/quality_gates/` directory currently contains only `__init__.py`.

---

## 9. Risks / Notes for Next Milestone

- The **real Sephora dataset** was ingested for the committed artifacts, while unit tests exercise the **synthetic fallback** — both paths are schema-identical, keeping CI hermetic.
- FAISS index currently uses a deterministic hashing embedder; runtime semantic retrieval will replace this with `nomic-embed-text` (Milestone 3).
- Artifacts are committed as read-only and must not be mutated at runtime — a key invariant for all downstream milestones.
