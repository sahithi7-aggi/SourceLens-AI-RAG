from urllib.parse import urlparse

import requests
from langchain_community.document_loaders import UnstructuredURLLoader


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36 SourceLensAI/1.0"
)


def normalize_urls(urls):
    """Clean, validate and deduplicate HTTP(S) URLs."""
    cleaned = []
    seen = set()

    for raw_url in urls:
        url = raw_url.strip()
        if not url:
            continue

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue

        if url not in seen:
            cleaned.append(url)
            seen.add(url)

    return cleaned


def _explain_exception(exc):
    """Turn common loader errors into user-friendly messages."""
    message = str(exc).strip()
    lower = message.lower()

    if "403" in lower or "forbidden" in lower:
        return "Access denied (HTTP 403). The website may block automated readers."
    if "404" in lower or "not found" in lower:
        return "Page not found (HTTP 404). Check that the URL is correct."
    if "401" in lower or "unauthorized" in lower:
        return "Authentication required (HTTP 401). This page may be private or login-protected."
    if "429" in lower or "too many requests" in lower:
        return "The website is rate-limiting requests (HTTP 429). Try again later."
    if "timeout" in lower or "timed out" in lower:
        return "The website took too long to respond."
    if "ssl" in lower or "certificate" in lower:
        return "The secure connection could not be established because of an SSL/certificate problem."
    if "connection" in lower or "max retries" in lower:
        return "Could not connect to the website. It may be unavailable or blocking automated requests."

    return f"The page could not be extracted: {message or 'unknown extraction error'}"


def _preflight_url(url):
    """Check whether a page is reachable before extraction."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
            allow_redirects=True,
        )
    except requests.exceptions.Timeout:
        return False, "The website took too long to respond."
    except requests.exceptions.SSLError:
        return False, "SSL/certificate verification failed."
    except requests.exceptions.RequestException as exc:
        return False, f"Could not connect to the website: {exc}"

    status = response.status_code

    if status == 401:
        return False, "Authentication required (HTTP 401)."
    if status == 403:
        return False, "Access denied (HTTP 403). The website may block automated readers."
    if status == 404:
        return False, "Page not found (HTTP 404)."
    if status == 429:
        return False, "The website is rate-limiting requests (HTTP 429)."
    if status >= 400:
        return False, f"The website returned HTTP {status}."

    content_type = response.headers.get("content-type", "").lower()
    if content_type and not any(
        kind in content_type
        for kind in (
            "text/html",
            "application/xhtml",
            "text/plain",
            "application/pdf",
        )
    ):
        return False, f"The URL returned unsupported content type: {content_type}."

    if not response.content.strip():
        return False, "The server returned an empty response."

    return True, None


def load_urls(urls):
    """
    Load every URL independently.

    Returns:
        documents: successfully extracted documents
        errors: compatibility list for the existing UI
        source_results: detailed status for every submitted URL
    """
    documents = []
    errors = []
    source_results = []

    for url in normalize_urls(urls):
        ok, reason = _preflight_url(url)

        if not ok:
            result = {
                "url": url,
                "status": "failed",
                "documents": 0,
                "reason": reason,
            }
            source_results.append(result)
            errors.append({"url": url, "error": reason})
            continue

        try:
            loader = UnstructuredURLLoader(urls=[url])
            loaded = loader.load()

            if not loaded:
                reason = "The page was reachable, but no readable text could be extracted."
                source_results.append(
                    {
                        "url": url,
                        "status": "failed",
                        "documents": 0,
                        "reason": reason,
                    }
                )
                errors.append({"url": url, "error": reason})
                continue

            for document in loaded:
                document.metadata["source"] = url

            documents.extend(loaded)
            source_results.append(
                {
                    "url": url,
                    "status": "success",
                    "documents": len(loaded),
                    "reason": "Source loaded and text extracted successfully.",
                }
            )

        except Exception as exc:
            reason = _explain_exception(exc)
            source_results.append(
                {
                    "url": url,
                    "status": "failed",
                    "documents": 0,
                    "reason": reason,
                }
            )
            errors.append({"url": url, "error": reason})

    return documents, errors, source_results
