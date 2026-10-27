"""
textrank.py
-----------
From-scratch implementation of the TextRank algorithm for extractive
text summarization, using NetworkX (graph) and NLTK (NLP preprocessing).

Algorithm overview:
  1. Sentence tokenization  -- split document into sentences
  2. Word tokenization       -- split each sentence into words
  3. Preprocessing           -- lowercase, remove stopwords & punctuation
  4. Similarity matrix       -- cosine similarity between sentence TF vectors
  5. Graph construction      -- sentences as nodes, similarity as edge weight
  6. PageRank                -- rank sentences using NetworkX's pagerank()
  7. Extraction              -- pick top-N sentences (by score), restore order

Reference:
  Mihalcea & Tarau (2004) "TextRank: Bringing Order into Texts"
"""

import math
import string
import re

import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# NLTK data download (runs once; silent after that)
# ---------------------------------------------------------------------------

def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    required = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
    }
    for path, pkg in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_STOP_EN = set(stopwords.words("english"))


def _clean_words(sentence: str) -> list:
    """
    Tokenize a sentence, lowercase, remove punctuation and stopwords.
    Returns a list of meaningful word tokens.
    """
    tokens = word_tokenize(sentence.lower())
    clean = [
        w for w in tokens
        if w not in string.punctuation
        and w not in _STOP_EN
        and re.match(r"^[a-z]+$", w)  # keep alphabetic only
    ]
    return clean


def _build_tf_vector(words: list) -> dict:
    """
    Build a simple term-frequency (TF) vector for a list of words.
    Returns a dict {word: tf_count}.
    """
    vec = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    return vec


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Compute cosine similarity between two TF vectors.
    Similarity = dot(A, B) / (||A|| * ||B||)
    Returns 0.0 if either vector is empty.
    """
    if not vec_a or not vec_b:
        return 0.0

    shared = set(vec_a) & set(vec_b)
    dot = sum(vec_a[w] * vec_b[w] for w in shared)

    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Core TextRank function
# ---------------------------------------------------------------------------

def textrank_summarize(text: str, ratio: float = 0.3) -> dict:
    """
    Summarize *text* using the TextRank algorithm.

    Parameters
    ----------
    text  : str   -- raw input text (English)
    ratio : float -- fraction of sentences to keep (0.10 to 0.50)

    Returns
    -------
    dict with keys:
        "summary"        : str  -- the extracted summary
        "original_sents" : int  -- total sentences in input
        "summary_sents"  : int  -- sentences kept
        "scores"         : list -- [(sentence, score), ...] sorted desc by score
        "error"          : str | None
    """
    # Validate ratio
    ratio = max(0.10, min(0.50, ratio))

    # Step 1: Sentence tokenization
    sentences = sent_tokenize(text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    n = len(sentences)

    if n == 0:
        return {
            "summary": "",
            "original_sents": 0,
            "summary_sents": 0,
            "scores": [],
            "error": "No valid sentences found in the input."
        }

    if n == 1:
        return {
            "summary": sentences[0],
            "original_sents": 1,
            "summary_sents": 1,
            "scores": [(sentences[0], 1.0)],
            "error": None
        }

    # Step 2 & 3: Tokenize and preprocess each sentence
    word_lists = [_clean_words(s) for s in sentences]
    tf_vectors = [_build_tf_vector(wl) for wl in word_lists]

    # Step 4: Build n x n similarity matrix
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = _cosine_similarity(tf_vectors[i], tf_vectors[j])

    # Step 5: Build weighted undirected graph
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            weight = sim_matrix[i][j]
            if weight > 0:
                graph.add_edge(i, j, weight=weight)

    # Step 6: PageRank (damping=0.85, standard TextRank setting)
    try:
        scores = nx.pagerank(graph, alpha=0.85, max_iter=100, weight="weight")
    except Exception:
        # Resilient fallback: pure Python PageRank power iteration
        damping = 0.85
        scores = {i: 1.0 / n for i in range(n)}
        for _ in range(40):
            next_scores = {}
            for i in range(n):
                in_sum = 0.0
                for j in graph.neighbors(i):
                    total_out = sum(graph[j][k].get("weight", 1.0) for k in graph.neighbors(j))
                    if total_out > 0:
                        in_sum += scores[j] * (graph[j][i].get("weight", 1.0) / total_out)
                next_scores[i] = (1.0 - damping) / n + damping * in_sum
            scores = next_scores

    # Step 7: Select top sentences by ratio, restore document order
    num_to_keep = max(1, round(n * ratio))
    ranked_indices = sorted(scores, key=lambda i: scores[i], reverse=True)
    selected_indices = sorted(ranked_indices[:num_to_keep])

    summary = " ".join(sentences[i] for i in selected_indices)

    score_list = sorted(
        [(sentences[i], round(scores[i], 6)) for i in range(n)],
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "summary": summary,
        "original_sents": n,
        "summary_sents": num_to_keep,
        "scores": score_list,
        "error": None,
    }
