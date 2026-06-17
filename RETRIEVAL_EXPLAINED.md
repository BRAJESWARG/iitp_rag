# Retrieval Deep Dive — BM25, Hybrid Search & RRF

How **Stage 1** of the two-stage pipeline actually finds candidate chunks: the
BM25 keyword retriever, its scoring formula, and how it's fused with the vector
(semantic) retriever via Reciprocal Rank Fusion. Companion to
[CODE_EXPLAINED.md](CODE_EXPLAINED.md) and [PROJECT_DETAILS.md](PROJECT_DETAILS.md).

---

## 1. What is the BM25 retriever?

**BM25 (Best Match 25)** is a classic **keyword / lexical** ranking algorithm — it
scores a document by how well its *words* match the query's words. It's one of the
two retrievers in Stage 1 (the other is the dense **vector** retriever).

| | BM25 (lexical) | Vector (semantic) |
|---|----------------|-------------------|
| Matches on | exact words / terms | meaning / embeddings |
| Great at | names, IDs, acronyms, technical terms | synonyms, paraphrases, concepts |
| Blind spot | synonyms ("car" ≠ "automobile") | exact rare tokens |
| Cost | very fast, no model | needs an embedding model |

They have **opposite trade-offs**, which is why the pipeline runs both and fuses
them — neither alone gives high recall.

**In code:** [retriever.py → `build_bm25_retriever()`](two_stage_rag/retriever.py) uses
LangChain's `BM25Retriever.from_documents(chunks)` (backed by the `rank-bm25`
library), with `k = STAGE1_K = 100`. It tokenizes the chunk texts and builds an
in-memory inverted index — no embeddings involved.

---

## 2. The BM25 scoring formula

For a query **Q** = {q₁ … qₙ} and document **D**, BM25 sums a per-term score:

```
                    n              f(qᵢ, D) · (k₁ + 1)
score(D, Q) =  Σ  IDF(qᵢ) · ─────────────────────────────────────
                   i=1         f(qᵢ, D) + k₁ · (1 − b + b · |D|/avgdl)
```

with the inverse document frequency:

```
IDF(qᵢ) = ln( 1 + (N − n(qᵢ) + 0.5) / (n(qᵢ) + 0.5) )
```

| Symbol | Meaning |
|--------|---------|
| `f(qᵢ, D)` | times term qᵢ appears in D (term frequency) |
| `|D|`, `avgdl` | length of D, and average doc length in the corpus |
| `N`, `n(qᵢ)` | total docs, and docs containing qᵢ |
| `k₁` | TF-saturation knob (rank-bm25 default **1.5**) |
| `b` | length-normalization knob (default **0.75**) |

### What each part does
- **IDF** — rare words (high `IDF`) count more than common ones. "InsightAI" ≫ "the".
- **TF saturation (k₁)** — a term's contribution rises with frequency but tops out
  at `IDF·(k₁+1)`: the 1st occurrence matters a lot, the 10th barely moves it.
  Smaller `k₁` saturates faster; larger `k₁` is more linear.
- **Length normalization (b)** — the `(1 − b + b·|D|/avgdl)` factor penalizes docs
  *longer* than average (a match in a 50-word chunk beats the same match in a
  5,000-word one). `b=0` disables it; `b=1` is full normalization.

Your stack uses `rank-bm25`'s **BM25Okapi** variant (k₁=1.5, b=0.75; it also floors
negative IDF for ultra-common terms).

**Tiny intuition:** for query `"BM25 algorithm"`, the chunk titled *"BM25: Best Match
25"* has high `f` for the rare, high-IDF term "BM25" **and** is short → big score. A
long transformer chunk that says "algorithm" once scores low. That's why BM25 alone
nails keyword queries.

---

## 3. Hybrid search & Reciprocal Rank Fusion (RRF)

**The problem:** BM25 scores (`~8.9`) and cosine similarities (`~0.87`) live on
completely different scales — you can't just add them.

**The solution:** [`EnsembleRetriever`](two_stage_rag/retriever.py) ignores raw scores
and fuses by **rank** using Reciprocal Rank Fusion:

```
                          1
RRF(d) =  Σ  wᵣ · ─────────────────
          r          c + rankᵣ(d)
```

| Symbol | Meaning |
|--------|---------|
| `rankᵣ(d)` | d's 1-based position in retriever r's list (1 = top) |
| `c` | smoothing constant (LangChain default **60**) |
| `wᵣ` | retriever weight → **[0.5, 0.5]** here (BM25, vector) |

### Worked example
A chunk ranked **#1 in BM25** and **#3 in vector**:
```
RRF = 0.5·1/(60+1) + 0.5·1/(60+3)
    = 0.00820       + 0.00794      = 0.01614
```
A chunk that's **#1 in BM25 only** (absent from vector's list) gets just `0.00820`.

### Key properties
- **Agreement wins** — docs ranked high by *both* retrievers float to the top.
- **Recall preserved** — a doc in only one list still gets a partial score and survives.
- **Scale-free** — only ranks are compared, never the incomparable raw scores.
- **`c = 60` dampens** rank gaps (rank 1 vs 2 isn't a huge difference) → robust fusion.
- The Ensemble then **dedupes by content** and returns the merged candidate list.

---

## 4. Live example on the sample corpus

Query: **"How does keyword-based document ranking work?"** over `sample.txt`
(66 chunks), printing each retriever's rank and the fused RRF score:

```
 RRF score | BM25 rk | Vec rk | chunk
------------------------------------------------------------------------
   0.01639 |    1    |   1    | BM25: Best Match 25 ...
   0.01454 |   13    |   5    | Stage 1 (High Recall - Bi-Encoder) ...
   0.01449 |    9    |   9    | Cross Encoder ...
   0.01387 |    5    |  21    | Cross Encoders vs Bi-Encoders ...
   0.01318 |   19    |  13    | Retrieval-Augmented Generation (RAG) ...
```

**Decoding it:**
- **Row 1** — ranked **#1 by both** → `0.5/61 + 0.5/61 = 0.01639`. Agreement → top candidate.
- **Row 2** — **vector-favored** (vec #5 vs BM25 #13): a semantic match without the exact keyword.
- **Row 4** — **BM25-favored** (BM25 #5 vs vec #21): strong keyword overlap the embedding ranked lower.
- Either signal alone would rank rows 2 & 4 very differently; RRF blends them so both survive.
- Note we never compared `8.9` to `0.87` — only ranks.

### The script
```python
"""Show BM25 rank vs vector rank vs fused RRF score for one query.
Run from two_stage_rag/:  PYTHONPATH=. venv/bin/python rrf_demo.py
"""
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from pipeline import create_pipeline

QUERY = "How does keyword-based document ranking work?"

p = create_pipeline(file_paths=None)
ens = p.retriever                      # EnsembleRetriever
bm25, vector = ens.retrievers          # [BM25Retriever, VectorStoreRetriever]
w_bm25, w_vec = ens.weights            # [0.5, 0.5]
c = getattr(ens, "c", 60)              # RRF smoothing constant

bm25_docs, vec_docs = bm25.invoke(QUERY), vector.invoke(QUERY)

def rank_map(docs):
    m = {}
    for i, d in enumerate(docs):
        m.setdefault(d.page_content, i + 1)   # 1-based rank
    return m

rb, rv = rank_map(bm25_docs), rank_map(vec_docs)
seen, order = set(), []
for d in bm25_docs + vec_docs:
    if d.page_content not in seen:
        seen.add(d.page_content); order.append(d.page_content)

rows = []
for content in order:
    a, b = rb.get(content), rv.get(content)
    score = (w_bm25 / (c + a) if a else 0) + (w_vec / (c + b) if b else 0)
    rows.append((score, a, b, content[:50].replace("\n", " ")))

rows.sort(key=lambda x: x[0], reverse=True)
print(f"{'RRF score':>10} | {'BM25 rk':>7} | {'Vec rk':>6} | chunk")
for score, a, b, snip in rows[:10]:
    print(f"{score:>10.5f} | {str(a):>7} | {str(b):>6} | {snip}")
```

---

## 5. How this feeds Stage 2

RRF gives a high-**recall** candidate set, but its scores are just rank-fusion
artifacts. So **Stage 2** re-scores those ~20–100 candidates with the
**cross-encoder** ([reranker.py](two_stage_rag/reranker.py)) — a model whose scores
*are* comparable (the `8.93 / −11` relevance logits) — and keeps the true top-5.

**Full Stage-1→2 flow:** BM25 (lexical ranks) + vector (semantic ranks)
→ **RRF** merge for recall → **cross-encoder** rerank for precision → top-5 → LLM.
