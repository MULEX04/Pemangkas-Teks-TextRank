/**
 * textrank.js
 * -----------
 * Pure JavaScript port of textrank.py
 * Implements TextRank extractive summarization entirely from scratch.
 *
 * Mirrors textrank.py (Python / NLTK / NetworkX) as closely as possible:
 *   Step 1 — Sentence tokenization  : abbreviation-aware regex splitter
 *   Step 2 — Word tokenization      : split on non-alpha chars
 *   Step 3 — Preprocessing          : lowercase, keep ^[a-z]+$, remove stopwords
 *   Step 4 — Similarity matrix      : cosine similarity of TF vectors
 *   Step 5 — Graph construction     : adjacency matrix (sentences = nodes)
 *   Step 6 — PageRank               : power iteration, alpha=0.85, max_iter=100
 *   Step 7 — Extraction             : top-N by score, restored to document order
 *
 * Reference:
 *   Mihalcea & Tarau (2004) "TextRank: Bringing Order into Texts"
 */

(function (global) {
    'use strict';

    // -------------------------------------------------------------------------
    // NLTK English stopwords — verbatim copy of stopwords.words("english")
    // Source: nltk_data/corpora/stopwords/english  (179 tokens)
    // -------------------------------------------------------------------------
    const STOP_EN = new Set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', "she's", 'her', 'hers', 'herself',
        'it', "it's", 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing',
        'a', 'an', 'the',
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', "don't",
        'should', "should've", 'now',
        'd', 'll', 'm', 'o', 're', 've', 'y',
        'ain',
        'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't",
        'hadn', "hadn't", 'hasn', "hasn't",
        'haven', "haven't", 'isn', "isn't",
        'ma', 'mightn', "mightn't", 'mustn', "mustn't",
        'needn', "needn't", 'shan', "shan't",
        'shouldn', "shouldn't", 'wasn', "wasn't",
        'weren', "weren't", 'won', "won't",
        'wouldn', "wouldn't",
    ]);

    // -------------------------------------------------------------------------
    // Step 1 — Sentence tokenizer
    // Mirrors: nltk.tokenize.sent_tokenize(text) for English
    //
    // Strategy:
    //   a) Temporarily replace periods in known abbreviations so they are not
    //      treated as sentence boundaries (same heuristic as NLTK's Punkt).
    //   b) Temporarily replace periods inside decimal numbers.
    //   c) Split on [.!?] followed by whitespace + capital letter / quote.
    //   d) Restore replaced periods.
    // -------------------------------------------------------------------------
    const _ABBREVS = [
        'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc',
        'Inc', 'Ltd', 'Corp', 'St', 'Ave', 'Blvd', 'Dept', 'Fig',
        'No', 'Vol', 'pp', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul',
        'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'al', 'approx', 'est',
        'govt', 'intl', 'natl', 'dept', 'temp', 'univ', 'assn',
        'Bros', 'Co', 'Corp', 'e.g', 'i.e',
    ];

    // Precompile abbreviation regexes once (case-sensitive to match Python)
    const _ABBREV_RES = _ABBREVS.map(a => ({
        re: new RegExp('\\b(' + a.replace('.', '\\.') + ')\\.', 'g'),
        placeholder: '$1\x00',
    }));

    function sentTokenize(text) {
        let t = text;

        // (a) Protect abbreviations: "Dr." → "Dr\x00"
        _ABBREV_RES.forEach(({ re, placeholder }) => {
            t = t.replace(re, placeholder);
        });

        // (b) Protect decimals: "3.14" → "3\x01\x0114"
        t = t.replace(/(\d)\.(\d)/g, '$1\x01$2');

        // (c) Split on sentence-ending punctuation (.  !  ?)
        //     followed by whitespace and a capital letter / opening quote
        //     (lookbehind supported in all modern browsers)
        const parts = t.split(/(?<=[.!?]["\u2019\u201d]?)\s+(?=[A-Z"'\u2018\u201c(])/);

        // (d) Restore placeholders and trim
        return parts
            .map(s => s.replace(/\x00/g, '.').replace(/\x01/g, '.').trim())
            .filter(s => s.length > 0);
    }

    // -------------------------------------------------------------------------
    // Steps 2 & 3 — Word tokenization + preprocessing
    // Mirrors: word_tokenize(sentence.lower()) + filter(^[a-z]+$, not stopword)
    //
    // In Python:
    //   tokens = word_tokenize(sentence.lower())
    //   clean  = [w for w in tokens
    //               if w not in string.punctuation   # redundant given next check
    //               and w not in _STOP_EN
    //               and re.match(r"^[a-z]+$", w)]    # ← decisive filter
    //
    // Because the decisive filter is ^[a-z]+$ (alphabetic only), the exact
    // tokenizer does not matter — splitting on any non-alpha character gives
    // the same set of surviving tokens.
    // -------------------------------------------------------------------------
    function cleanWords(sentence) {
        return sentence
            .toLowerCase()
            .split(/[^a-z]+/)                          // split on non-alpha chars
            .filter(w => w.length > 0               // remove empty strings
                      && /^[a-z]+$/.test(w)          // alphabetic only (= Python filter)
                      && !STOP_EN.has(w));            // remove stopwords
    }

    // -------------------------------------------------------------------------
    // Step 3b — TF vector
    // Mirrors: _build_tf_vector(words) → {word: count}
    // -------------------------------------------------------------------------
    function buildTfVector(words) {
        const vec = {};
        words.forEach(w => { vec[w] = (vec[w] || 0) + 1; });
        return vec;
    }

    // -------------------------------------------------------------------------
    // Step 4 — Cosine similarity between two TF vectors
    // Mirrors: _cosine_similarity(vec_a, vec_b)
    //   sim = dot(A,B) / (||A|| * ||B||)
    // -------------------------------------------------------------------------
    function cosineSimilarity(vecA, vecB) {
        const keysA = Object.keys(vecA);
        const keysB = Object.keys(vecB);
        if (keysA.length === 0 || keysB.length === 0) return 0.0;

        const setB = new Set(keysB);
        let dot = 0;
        keysA.forEach(w => { if (setB.has(w)) dot += vecA[w] * vecB[w]; });

        const magA = Math.sqrt(keysA.reduce((s, w) => s + vecA[w] ** 2, 0));
        const magB = Math.sqrt(keysB.reduce((s, w) => s + vecB[w] ** 2, 0));

        if (magA === 0 || magB === 0) return 0.0;
        return dot / (magA * magB);
    }

    // -------------------------------------------------------------------------
    // Steps 5 & 6 — Weighted PageRank
    // Mirrors: nx.pagerank(G, alpha=0.85, max_iter=100, weight='weight')
    //
    // NetworkX implementation (power-iteration, weighted undirected graph):
    //   W[i][j] = edge_weight(i,j) / Σ_k edge_weight(i,k)   (row-stochastic)
    //   dangling nodes → no outgoing weight; PR distributed uniformly
    //
    //   For each iteration:
    //     dangleSum  = alpha * Σ_{d∈dangling} prev[d]
    //     scores[i]  = alpha * Σ_j(prev[j] * W[j][i])
    //                + dangleSum * (1/N)
    //                + (1 - alpha) * (1/N)
    //
    //   Convergence: Σ|scores[i] - prev[i]| < N * tol   (tol = 1e-6)
    //   Fallback (no convergence): return uniform scores  (= Python fallback)
    //
    // @param {number[][]} adjMatrix  n×n similarity matrix (symmetric)
    // @param {number}     n
    // @param {number}     alpha      damping factor            (default 0.85)
    // @param {number}     maxIter    max power iterations      (default 100)
    // @param {number}     tol        convergence tolerance     (default 1e-6)
    // @returns {number[]} PageRank score per node
    // -------------------------------------------------------------------------
    function pagerank(adjMatrix, n, alpha = 0.85, maxIter = 100, tol = 1e-6) {
        // Build row-stochastic weight matrix W
        const W = Array.from({ length: n }, () => new Array(n).fill(0.0));
        const danglingNodes = [];

        for (let i = 0; i < n; i++) {
            const rowSum = adjMatrix[i].reduce((a, b) => a + b, 0);
            if (rowSum === 0) {
                danglingNodes.push(i);           // isolated node — no outgoing weight
            } else {
                for (let j = 0; j < n; j++) {
                    W[i][j] = adjMatrix[i][j] / rowSum;
                }
            }
        }

        const p = 1.0 / n;                      // uniform personalization / teleport
        let scores = new Array(n).fill(p);

        for (let iter = 0; iter < maxIter; iter++) {
            const prev = scores.slice();

            // Dangling node contribution: alpha * Σ PR(dangling) * (1/N)
            const dangleContrib = alpha
                * danglingNodes.reduce((s, d) => s + prev[d], 0)
                * p;

            for (let i = 0; i < n; i++) {
                // Weighted in-flow from all neighbours
                let inflow = 0;
                for (let j = 0; j < n; j++) {
                    inflow += prev[j] * W[j][i];
                }
                scores[i] = alpha * inflow + dangleContrib + (1.0 - alpha) * p;
            }

            // Convergence check: Σ|scores[i] - prev[i]| < N * tol
            const err = scores.reduce((s, si, i) => s + Math.abs(si - prev[i]), 0);
            if (err < n * tol) break;
            // If not converged after maxIter → keep last scores (= Python fallback behaviour)
        }

        return scores;
    }

    // -------------------------------------------------------------------------
    // Public API — textrankSummarize(text, ratio)
    // Mirrors: textrank_summarize(text, ratio) in textrank.py
    //
    // @param {string} text   Raw English input text
    // @param {number} ratio  Fraction of sentences to keep (0.10–0.50)
    // @returns {Object}
    //   { summary, originalSents, summarySents, scores, error }
    // -------------------------------------------------------------------------
    function textrankSummarize(text, ratio = 0.3) {
        // Clamp ratio — mirrors: ratio = max(0.10, min(0.50, ratio))
        ratio = Math.max(0.10, Math.min(0.50, ratio));

        // Step 1: Sentence tokenization
        let sentences = sentTokenize(text.trim());
        sentences = sentences.filter(s => s.trim().length > 10);   // same as Python

        const n = sentences.length;

        if (n === 0) {
            return {
                summary: '', originalSents: 0, summarySents: 0,
                scores: [], error: 'No valid sentences found in the input.',
            };
        }
        if (n === 1) {
            return {
                summary: sentences[0], originalSents: 1, summarySents: 1,
                scores: [[sentences[0], 1.0]], error: null,
            };
        }

        // Steps 2 & 3: Preprocess sentences
        const wordLists = sentences.map(s => cleanWords(s));
        const tfVectors = wordLists.map(wl => buildTfVector(wl));

        // Step 4: n×n cosine similarity matrix
        const simMatrix = Array.from({ length: n }, () => new Array(n).fill(0.0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    simMatrix[i][j] = cosineSimilarity(tfVectors[i], tfVectors[j]);
                }
            }
        }

        // Steps 5 & 6: Weighted PageRank
        // (graph construction is implicit — adjMatrix IS the weighted graph)
        const prScores = pagerank(simMatrix, n, 0.85, 100, 1e-6);

        // Step 7: Select top-N sentences, restore document order
        // Mirrors: ranked_indices = sorted(scores, key=..., reverse=True)
        //          selected_indices = sorted(ranked_indices[:num_to_keep])
        const numToKeep = Math.max(1, Math.round(n * ratio));

        const rankedIndices = prScores
            .map((s, i) => [i, s])
            .sort((a, b) => b[1] - a[1])       // descending by score
            .map(([i]) => i);

        const selectedIndices = rankedIndices
            .slice(0, numToKeep)
            .sort((a, b) => a - b);             // restore document order

        const summary = selectedIndices.map(i => sentences[i]).join(' ');

        // Score list sorted descending (mirrors Python's score_list)
        const scoreList = prScores
            .map((s, i) => [sentences[i], parseFloat(s.toFixed(6))])
            .sort((a, b) => b[1] - a[1]);

        return {
            summary,
            originalSents: n,
            summarySents:  numToKeep,
            scores:        scoreList,
            error:         null,
        };
    }

    // Expose to global scope (browser window)
    global.textrankSummarize = textrankSummarize;

})(typeof window !== 'undefined' ? window : global);
