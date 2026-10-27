"""
scraper.py
----------
Modul universal article scraper untuk mengekstrak teks lengkap dan metadata
dari berbagai website artikel/berita menggunakan trafilatura dengan
resilient HTML fallback parser.

Fungsi utama:
    scrape_article(url) -> dict
        Mengembalikan dict dengan key:
            "text"       : str         -- teks artikel lengkap & bersih
            "title"      : str         -- judul artikel
            "url"        : str         -- URL final artikel
            "domain"     : str         -- nama domain website
            "author"     : str         -- penulis (jika ditemukan)
            "date"       : str         -- tanggal publikasi (jika ditemukan)
            "word_count" : int         -- jumlah kata hasil ekstraksi
            "error"      : str | None  -- pesan error jika gagal
"""

from html.parser import HTMLParser
from urllib.parse import urlparse
import re
import requests
import trafilatura


# ──────────────────────────────────────────────────────────────────────────────
# Konfigurasi Network & Headers
# ──────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Upgrade-Insecure-Requests": "1",
}

REQUEST_TIMEOUT = 15  # detik

DISALLOWED_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp3", ".mp4", ".avi", ".mkv", ".mov",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".exe", ".msi", ".apk", ".bin",
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Validasi URL
# ──────────────────────────────────────────────────────────────────────────────

def validate_url(url: str) -> tuple[bool, str | None]:
    """
    Validasi format URL website artikel.
    Mengembalikan tuple (is_valid, error_message).
    """
    if not url or not url.strip():
        return False, "URL artikel tidak boleh kosong."

    clean_url = url.strip()

    if not clean_url.startswith(("http://", "https://")):
        return False, "URL harus diawali dengan http:// atau https://"

    try:
        parsed = urlparse(clean_url)
    except Exception:
        return False, "Format URL tidak valid."

    hostname = (parsed.hostname or "").lower()
    if not hostname or "." not in hostname:
        return False, "Format domain pada URL tidak valid (contoh: https://domain.com/artikel)."

    # Cek ekstensi file media/biner
    path_lower = parsed.path.lower()
    for ext in DISALLOWED_EXTENSIONS:
        if path_lower.endswith(ext):
            return False, f"URL mengarah ke file media/biner ('{ext}'), bukan halaman artikel web."

    return True, None


def get_domain_display(url: str) -> str:
    """Mengambil nama domain bersih untuk tampilan UI."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        return host.removeprefix("www.")
    except Exception:
        return "Website"


# ──────────────────────────────────────────────────────────────────────────────
# Resilient Fallback HTML Parser
# ──────────────────────────────────────────────────────────────────────────────

class _HtmlArticleFallbackExtractor(HTMLParser):
    """
    Parser HTML fallback jika trafilatura tidak dapat mengekstrak teks.
    Mengabaikan tag non-konten (script, style, nav, footer, header, dll.)
    dan mengumpulkan teks dari tag paragraf (<p>) dan judul (<h1-h6>).
    """

    IGNORE_TAGS = {
        "script", "style", "noscript", "svg", "nav", "footer",
        "header", "aside", "form", "button", "iframe", "head",
    }

    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "article", "section"}

    def __init__(self):
        super().__init__()
        self.ignore_depth = 0
        self.title_depth = 0
        self.title_text = []
        self.paragraphs = []
        self._current_block = []

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self.IGNORE_TAGS:
            self.ignore_depth += 1
        elif t == "title":
            self.title_depth += 1
        elif t in self.BLOCK_TAGS:
            self._flush_block()

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self.IGNORE_TAGS:
            if self.ignore_depth > 0:
                self.ignore_depth -= 1
        elif t == "title":
            if self.title_depth > 0:
                self.title_depth -= 1
        elif t in self.BLOCK_TAGS:
            self._flush_block()

    def handle_data(self, data):
        if self.ignore_depth > 0:
            return

        if self.title_depth > 0:
            self.title_text.append(data)
            return

        cleaned = re.sub(r"\s+", " ", data)
        if cleaned.strip():
            self._current_block.append(cleaned)

    def _flush_block(self):
        if self._current_block:
            combined = "".join(self._current_block).strip()
            # Hanya simpan blok yang memiliki setidaknya beberapa kata
            if len(combined.split()) >= 4:
                self.paragraphs.append(combined)
            self._current_block = []

    def get_result(self) -> tuple[str, str]:
        self._flush_block()
        title = "".join(self.title_text).strip()
        body = "\n\n".join(self.paragraphs)
        return title, body


def _fallback_extract(html: str) -> tuple[str, str]:
    """Ekstraksi teks darurat menggunakan _HtmlArticleFallbackExtractor."""
    try:
        parser = _HtmlArticleFallbackExtractor()
        parser.feed(html)
        title, text = parser.get_result()
        return title, text
    except Exception:
        return "", ""


# ──────────────────────────────────────────────────────────────────────────────
# Fungsi Utama Scraping
# ──────────────────────────────────────────────────────────────────────────────

def scrape_article(url: str) -> dict:
    """
    Scrape dan ekstrak seluruh teks artikel dari URL website apapun.

    Parameters
    ----------
    url : str -- URL artikel dari situs berita atau blog manapun

    Returns
    -------
    dict dengan struktur:
        {
            "text": str,
            "title": str,
            "url": str,
            "domain": str,
            "author": str,
            "date": str,
            "word_count": int,
            "error": str | None
        }
    """
    clean_url = (url or "").strip()

    # ── 1. Validasi URL ──────────────────────────────────────────────────────
    is_valid, err = validate_url(clean_url)
    if not is_valid:
        return {
            "text": "",
            "title": "",
            "url": clean_url,
            "domain": "",
            "author": "",
            "date": "",
            "word_count": 0,
            "error": err,
        }

    domain = get_domain_display(clean_url)

    # ── 2. Unduh Halaman Web ─────────────────────────────────────────────────
    html_content = None
    final_url = clean_url

    try:
        response = requests.get(
            clean_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()
        final_url = response.url
        html_content = response.text
    except requests.exceptions.Timeout:
        return {
            "text": "", "title": "", "url": clean_url, "domain": domain,
            "author": "", "date": "", "word_count": 0,
            "error": "Koneksi timeout saat mengakses website. Periksa koneksi internet atau coba beberapa saat lagi.",
        }
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "?"
        if status_code == 403:
            msg = f"Akses ditolak oleh website ({domain} HTTP 403). Website ini membatasi bot/scraper otomatis."
        elif status_code == 404:
            msg = f"Halaman artikel tidak ditemukan (HTTP 404). Pastikan URL artikel masih aktif."
        else:
            msg = f"Gagal mengakses halaman artikel (HTTP {status_code})."
        return {
            "text": "", "title": "", "url": clean_url, "domain": domain,
            "author": "", "date": "", "word_count": 0,
            "error": msg,
        }
    except requests.exceptions.RequestException as e:
        # Fallback coba fetch melalui trafilatura.fetch_url jika requests biasa gagal
        try:
            downloaded = trafilatura.fetch_url(clean_url)
            if downloaded:
                html_content = downloaded
            else:
                return {
                    "text": "", "title": "", "url": clean_url, "domain": domain,
                    "author": "", "date": "", "word_count": 0,
                    "error": f"Gagal memuat URL: {str(e)}",
                }
        except Exception:
            return {
                "text": "", "title": "", "url": clean_url, "domain": domain,
                "author": "", "date": "", "word_count": 0,
                "error": f"Gagal menghubungkan ke website ({domain}): {str(e)}",
            }

    if not html_content or not html_content.strip():
        return {
            "text": "", "title": "", "url": final_url, "domain": domain,
            "author": "", "date": "", "word_count": 0,
            "error": "Konten halaman web kosong.",
        }

    # ── 3. Ekstraksi Menggunakan Trafilatura ──────────────────────────────────
    extracted_text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
        deduplicate=True,
        url=final_url,
    )

    # Ekstraksi metadata artikel
    meta = trafilatura.extract_metadata(html_content, default_url=final_url)
    title = meta.title if (meta and meta.title) else ""
    author = meta.author if (meta and meta.author) else ""
    date = meta.date if (meta and meta.date) else ""

    # ── 4. Fallback Extraction jika Trafilatura kosong/sangat pendek ──────────
    if not extracted_text or len(extracted_text.strip().split()) < 30:
        fb_title, fb_text = _fallback_extract(html_content)
        if fb_text and len(fb_text.split()) > len((extracted_text or "").split()):
            extracted_text = fb_text
            if not title and fb_title:
                title = fb_title

    if not extracted_text or len(extracted_text.strip().split()) < 20:
        return {
            "text": "",
            "title": title or f"Artikel {domain}",
            "url": final_url,
            "domain": domain,
            "author": author,
            "date": date,
            "word_count": 0,
            "error": (
                f"Tidak dapat mengekstrak isi teks artikel dari {domain}. "
                "Halaman mungkin menggunakan rendering JavaScript penuh (SPA), memerlukan login, "
                "atau bukan merupakan halaman artikel berita teks."
            ),
        }

    clean_text = extracted_text.strip()
    word_count = len(clean_text.split())

    return {
        "text": clean_text,
        "title": title or f"Artikel ({domain})",
        "url": final_url,
        "domain": domain,
        "author": author,
        "date": date,
        "word_count": word_count,
        "error": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Kompatibilitas ke belakang (Backward Compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def scrape_bbc(url: str) -> dict:
    """
    Alias scrape_bbc untuk menjaga kompatibilitas jika dipanggil dari kode lama.
    Sekarang dapat memproses URL BBC maupun URL website lainnya.
    """
    return scrape_article(url)
