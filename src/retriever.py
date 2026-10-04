from urllib.parse import urlparse


def source_label(url):
    """Return a compact hostname for display."""
    host = urlparse(url).netloc.replace("www.", "")
    return host or url


def _clamp_score(score):
    """Keep relevance scores display-safe."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(1.0, value))


def _unpack_result(item):
    """
    Normalize the result shape returned by Chroma/LangChain.

    The normal shape is (Document, score), but this keeps the app
    tolerant of equivalent tuple/list result shapes.
    """
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return item[0], item[1]

    raise TypeError(
        f"Unexpected similarity result format: {type(item).__name__}"
    )


def _format_results(pairs):
    results = []

    for item in pairs:
        document, score = _unpack_result(item)
        results.append(
            {
                "document": document,
                "score": _clamp_score(score),
            }
        )

    return results


def _retrieval_query(question):
    """Add lightweight domain terms that improve semantic retrieval for tool/function questions."""
    q = question.strip()
    lowered = q.lower()

    if any(term in lowered for term in ("tool calling", "function calling", "function/tool", "tools")):
        return (
            f"{q} tool calling function calling tools API tool definitions "
            "schema parameters arguments execution results responses"
        )

    return q


def _search(vectorstore, question, k, source=None):
    """Use Chroma's raw distance search and convert distance to a stable display score."""
    query = _retrieval_query(question)

    kwargs = {"k": k}
    if source is not None:
        kwargs["filter"] = {"source": source}

    pairs = vectorstore.similarity_search_with_score(query, **kwargs)

    results = []
    for document, distance in pairs:
        # Chroma's score here is a distance: lower is better. This conversion
        # is only for UI display; retrieval ordering remains Chroma's ordering.
        try:
            distance = max(0.0, float(distance))
        except (TypeError, ValueError):
            distance = 1.0

        relevance = 1.0 / (1.0 + distance)
        results.append({
            "document": document,
            "score": relevance,
        })

    return results


def retrieve_documents(vectorstore, question, k=5):
    """Normal mode: retrieve the globally most relevant chunks."""
    return _search(vectorstore, question, k)


def retrieve_per_source(vectorstore, question, sources, k_per_source=3):
    """
    Compare mode: retrieve independently from every readable source.
    """
    results = []

    for source in sources:
        try:
            results.extend(_search(vectorstore, question, k_per_source, source))
        except Exception as exc:
            # Some Chroma/LangChain combinations reject filtered score search.
            # Fall back to a wider unfiltered search and retain this source only.
            pool = _search(vectorstore, question, max(k_per_source * len(sources), 20))
            source_results = [
                item for item in pool
                if item["document"].metadata.get("source") == source
            ][:k_per_source]
            results.extend(source_results)

    return _assign_source_numbers(results)

def _assign_source_numbers(results):
    """Assign one citation number to each unique source."""
    source_numbers = {}

    for item in results:
        source = item["document"].metadata.get("source", "Unknown source")

        if source not in source_numbers:
            source_numbers[source] = len(source_numbers) + 1

        item["source_number"] = source_numbers[source]

    return results


def build_context(results):
    """
    Build LLM context while ensuring chunks from the same URL
    share the same citation number.
    """
    source_numbers = {}
    context_parts = []

    for item in results:
        document = item["document"]
        source = document.metadata.get("source", "Unknown source")

        if source not in source_numbers:
            source_numbers[source] = len(source_numbers) + 1

        number = source_numbers[source]
        item["source_number"] = number

        context_parts.append(
            f"SOURCE [{number}]\n"
            f"URL: {source}\n"
            f"CONTENT:\n{document.page_content}"
        )

    return "\n\n---\n\n".join(context_parts), source_numbers
