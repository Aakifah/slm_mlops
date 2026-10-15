# Intent Router Reference Decision (2026-10-15)

Researched via web + GitHub API before cloning. Only the effective
implementations are pulled; stale/wrong-layer ones are cited, not vendored.

| Repo | Stars | Last code push | Verdict |
|---|---|---|---|
| `aurelio-labs/semantic-router` | ~3950 | 2026-10-15 | PRIMARY |
| `huggingface/setfit` | ~2840 | 2026-10-06 | BACKUP |
| `lm-sys/RouteLLM` | ~5600 | 2024-08-10 | EXCLUDED |

- **PRIMARY semantic-router:** binary simple/complex routing maps directly onto
  route + utterance lists with cosine similarity over `nomic-embed-text`
  (Ollama) embeddings. Zero training, config-only routes, actively maintained.
  First attempt at the >=92% accuracy gate.
- **BACKUP setfit:** actively maintained few-shot classifier (contrastive
  fine-tune + sklearn head, ~8 examples/class). Heavier ops (trains a
  checkpoint); use only if zero-shot routing misses the gate, persisting to
  `src/artifacts/models/intent_classifier.pkl`.
- **EXCLUDED RouteLLM:** solves learned strong-vs-weak LLM cost routing on
  human-preference data, not deterministic simple/complex classification, and
  the code is stale since 2024-08. Related work only; not cloned.
