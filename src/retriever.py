from urllib.parse import urlparse


def source_label(source):
    if source.startswith(("http://", "https://")):
        return urlparse(source).netloc.replace("www.", "") or source
    return source


def _search(vectorstore, question, k, source=None):
    kwargs = {"k": k}
    if source is not None:
        kwargs["filter"] = {"source": source}

    pairs = vectorstore.similarity_search_with_score(question, **kwargs)
    results = []

    for document, distance in pairs:
        try:
            distance = max(0.0, float(distance))
        except (TypeError, ValueError):
            distance = 1.0

        results.append({
            "document": document,
            "score": 1.0 / (1.0 + distance),
        })

    return results


def retrieve_documents(vectorstore, question, k=5):
    return _assign_source_numbers(_search(vectorstore, question, k))


def retrieve_per_source(vectorstore, question, sources, k_per_source=3):
    results = []
    for source in sources:
        try:
            results.extend(_search(vectorstore, question, k_per_source, source))
        except Exception:
            pool = _search(vectorstore, question, max(k_per_source * len(sources), 20))
            results.extend([
                item for item in pool
                if item["document"].metadata.get("source") == source
            ][:k_per_source])
    return _assign_source_numbers(results)


def _assign_source_numbers(results):
    source_numbers = {}
    for item in results:
        source = item["document"].metadata.get("source", "Unknown source")
        if source not in source_numbers:
            source_numbers[source] = len(source_numbers) + 1
        item["source_number"] = source_numbers[source]
    return results


def build_context(results):
    source_numbers = {}
    context_parts = []

    for item in results:
        document = item["document"]
        metadata = document.metadata
        source = metadata.get("source", "Unknown source")

        if source not in source_numbers:
            source_numbers[source] = len(source_numbers) + 1

        number = source_numbers[source]
        page = metadata.get("page")
        section = metadata.get("section")
        method = metadata.get("extraction_method")

        location = f"Source: {source}"
        if page:
            location += f" | Page: {page}"
        if section:
            location += f" | Section: {section}"
        if method:
            location += f" | Extraction: {method}"

        item["source_number"] = number

        context_parts.append(
            f"SOURCE [{number}]\n{location}\nCONTENT:\n{document.page_content}"
        )

    return "\n\n---\n\n".join(context_parts), source_numbers
