MODE_CONFIG = {
    "AI": {
        "description": "General AI and technology research.",
        "question": "What are the main points across these sources?",
        "examples": ["Summarize the main ideas and supporting evidence."],
    },
    "Product": {
        "description": "Product and feature research.",
        "question": "What are the important product capabilities?",
        "examples": ["What are the key features and limitations?"],
    },
    "Research Papers": {
        "description": "Research-oriented analysis.",
        "question": "What are the main findings?",
        "examples": ["What methodology and findings are described?"],
    },
    "Documentation": {
        "description": "Technical documentation analysis.",
        "question": "How does this work?",
        "examples": ["Explain the main workflow and important details."],
    },
    "Company Research": {
        "description": "Company and business research.",
        "question": "What are the important company details?",
        "examples": ["What are the key facts and risks?"],
    },
}


def hostname(url):
    from urllib.parse import urlparse
    if url.startswith(("http://", "https://")):
        return urlparse(url).netloc.replace("www.", "") or url
    return url
