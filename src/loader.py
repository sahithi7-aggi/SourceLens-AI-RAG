from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

TIMEOUT = 20


def normalize_urls(urls):
    """Clean, validate and deduplicate URLs."""
    normalized = []

    for url in urls:
        url = url.strip()

        if not url:
            continue

        if not url.startswith(("http://", "https://")):
            continue

        parsed = urlparse(url)

        if not parsed.netloc:
            continue

        if url not in normalized:
            normalized.append(url)

    return normalized


def _fetch_page(url):
    """Fetch a webpage and return its HTML."""
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=TIMEOUT,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        raise ValueError(
            f"Unsupported content type: {content_type or 'unknown'}"
        )

    return response.text, response.url


def _extract_text(html):
    """Extract readable text from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that generally contain navigation,
    # styling, scripts, or other non-content information.
    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
        ]
    ):
        element.decompose()

    # Prefer the main article/content area when available.
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
    )

    target = main if main else soup

    text = target.get_text(separator="\n", strip=True)

    # Remove excessive blank lines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return "\n".join(lines)


def _format_error(exc):
    """Return a user-friendly error message."""
    if isinstance(exc, requests.exceptions.Timeout):
        return "Request timed out while fetching the webpage."

    if isinstance(exc, requests.exceptions.TooManyRedirects):
        return "The webpage redirected too many times."

    if isinstance(exc, requests.exceptions.SSLError):
        return "SSL error while connecting to the webpage."

    if isinstance(exc, requests.exceptions.ConnectionError):
        return "Could not connect to the webpage."

    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None

        if status == 401:
            return "The webpage requires authentication (HTTP 401)."

        if status == 403:
            return "The webpage denied automated access (HTTP 403)."

        if status == 404:
            return "The webpage was not found (HTTP 404)."

        if status == 429:
            return "The webpage rate-limited the request (HTTP 429)."

        return f"The webpage returned HTTP {status}."

    return str(exc)


def load_urls(urls):
    """
    Load webpages into LangChain Documents.

    Returns:
        documents: successfully extracted Documents
        errors: dictionary of URL -> error message
        source_results: per-source processing status
    """
    documents = []
    errors = []
    source_results = []

    for url in normalize_urls(urls):
        try:
            html, final_url = _fetch_page(url)
            text = _extract_text(html)

            if not text or len(text.strip()) < 100:
                raise ValueError(
                    "The webpage did not contain enough readable text."
                )

            document = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "final_url": final_url,
                    "title": urlparse(final_url).netloc,
                },
            )

            documents.append(document)

            source_results.append(
                {
                    "url": url,
                    "status": "success",
                    "documents": 1,
                    "reason": "",
                }
            )

        except Exception as exc:
            reason = _format_error(exc)

            errors.append(
                {
                    "url": url,
                    "error": reason,
                }
            )

            source_results.append(
                {
                    "url": url,
                    "status": "failed",
                    "documents": 0,
                    "reason": reason,
                }
            )

    return documents, errors, source_results