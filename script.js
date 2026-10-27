/**
 * script.js
 * ---------
 * Frontend logic for the TextRank Summarizer (static / no-backend version).
 * Calls textrankSummarize() from textrank.js directly in the browser.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── Element references ────────────────────────────────────────────────
    const inputText      = document.getElementById('inputText');
    const wordCountBadge = document.getElementById('wordCountBadge');
    const wordCountText  = document.getElementById('wordCountText');
    const ratioSlider    = document.getElementById('ratioSlider');
    const ratioDisplay   = document.getElementById('ratioDisplay');
    const btnClear       = document.getElementById('btnClear');
    const btnCopy        = document.getElementById('btnCopy');
    const btnRingkas     = document.getElementById('btnRingkas');
    const outputText     = document.getElementById('outputText');
    const toast          = document.getElementById('toast');
    const toastMsg       = document.getElementById('toastMsg');
    const statsBar       = document.getElementById('statsBar');

    const MAX_WORDS = 3000;

    // ── Word counter ──────────────────────────────────────────────────────
    inputText.addEventListener('input', updateWordCount);

    function countWords(text) {
        return text.trim() ? text.trim().split(/\s+/).filter(w => w.length > 0).length : 0;
    }

    function updateWordCount() {
        const count = countWords(inputText.value);
        wordCountText.textContent = `${count} / ${MAX_WORDS} words`;
        wordCountBadge.classList.toggle('exceeded', count > MAX_WORDS);
    }

    // ── Slider ────────────────────────────────────────────────────────────
    ratioSlider.addEventListener('input', () => {
        ratioDisplay.textContent = `${ratioSlider.value}%`;
    });

    // ── Reset button ──────────────────────────────────────────────────────
    if (btnClear) {
        btnClear.addEventListener('click', () => {
            inputText.value  = '';
            outputText.value = '';
            if (statsBar) statsBar.classList.add('hidden');
            updateWordCount();
        });
    }

    // ── Copy button ───────────────────────────────────────────────────────
    if (btnCopy) {
        btnCopy.addEventListener('click', () => {
            const content = outputText.value.trim();
            if (!content) {
                showToast('Summary is empty!', 'warn');
                return;
            }
            navigator.clipboard.writeText(content).then(() => {
                showToast('Copied to clipboard!', 'success');
            });
        });
    }

    // ── Summarize button ──────────────────────────────────────────────────
    document.getElementById('summarizerForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await runSummarize();
    });

    async function runSummarize() {
        const text  = inputText.value.trim();
        const ratio = parseInt(ratioSlider.value, 10) / 100;

        // Client-side validation
        if (!text) {
            showToast('Please enter some text first.', 'warn');
            return;
        }
        if (countWords(text) > MAX_WORDS) {
            showToast(`Text exceeds ${MAX_WORDS} words.`, 'warn');
            return;
        }

        // UI: loading state
        setLoading(true);
        outputText.value = '';
        if (statsBar) statsBar.classList.add('hidden');

        // Yield to browser to render loading state before heavy computation
        await new Promise(resolve => setTimeout(resolve, 30));

        try {
            // Call TextRank directly in the browser (no server needed)
            const result = textrankSummarize(text, ratio);

            if (result.error) {
                showToast(result.error, 'warn');
                return;
            }

            // Display summary
            outputText.value = result.summary;

            // Display stats bar
            if (statsBar) {
                document.getElementById('statOriginal').textContent = result.originalSents;
                document.getElementById('statKept').textContent     = result.summarySents;
                document.getElementById('statRatio').textContent    = `${Math.round(ratio * 100)}%`;
                statsBar.classList.remove('hidden');
            }

            showToast('Summary generated!', 'success');

        } catch (err) {
            showToast('An error occurred. See console for details.', 'warn');
            console.error('TextRank error:', err);
        } finally {
            setLoading(false);
        }
    }

    // ── Loading state helper ──────────────────────────────────────────────
    function setLoading(isLoading) {
        btnRingkas.disabled = isLoading;
        const icon = btnRingkas.querySelector('i');
        const span = btnRingkas.querySelector('span');
        if (isLoading) {
            if (icon) icon.setAttribute('data-lucide', 'loader');
            if (span) span.textContent = 'Processing…';
            btnRingkas.classList.add('loading');
        } else {
            if (icon) icon.setAttribute('data-lucide', 'zap');
            if (span) span.textContent = 'Summarize';
            btnRingkas.classList.remove('loading');
        }
        lucide.createIcons();
    }

    // ── Toast notification ────────────────────────────────────────────────
    let toastTimer = null;
    function showToast(msg, type = 'success') {
        toastMsg.textContent = msg;
        toast.className = `toast ${type}`;
        toast.classList.remove('hidden');

        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            toast.classList.add('hidden');
        }, 2800);
    }
});
