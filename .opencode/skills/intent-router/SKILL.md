---
name: intent-router
description: Implement or refactor the FAISS semantic cache and reference-backed intent classifier. Use when touching intent_classifier.py, semantic_cache.py, or anything in references/.
---

# Intent Router

Load `llm-cost-optimization`, `llm-rag`, and `embedding-strategies` first, then apply these project rules.

## Rules

1. Zero regex or keyword heuristics in `src/model_customization/unified_router/intent_classifier.py`.
2. Ensure `bash scripts/pull_references.sh` has populated `references/semantic-router`, `references/RouteLLM`, `references/setfit` (gitignored except `references/README.md`).
3. Route single-record predictions to `QueryRouteType.SIMPLE_ML_BASELINE`, analytical queries to `QueryRouteType.COMPLEX_SLM_SQL`.
4. Cache hits require cosine similarity >= 0.95 on `nomic-embed-text` vectors and return validated `QueryExecutionResult` with `cached=True`.
5. Verify with `pytest tests/quality_gates/test_intent_routing.py -v`.
