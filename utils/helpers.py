from urllib.parse import urlparse


MODE_CONFIG = {
    "🤖 AI Research": {
        "description": "Compare AI platforms, models, APIs, agents and developer capabilities.",
        "question": "Compare the main approaches described by these AI sources.",
        "examples": [
            "How do these companies approach tool calling?",
            "What are the main differences between their agent architectures?",
            "Compare the capabilities described by each source.",
        ],
    },
    "🛒 Product Research": {
        "description": "Research products from the sources you provide and compare their specifications.",
        "question": "Compare the products described by these sources.",
        "examples": [
            "Which product is best for programming?",
            "Compare price, performance and key specifications.",
            "What are the major trade-offs?",
        ],
    },
    "📄 Research Papers": {
        "description": "Compare papers, approaches, findings, limitations and research directions.",
        "question": "What problem are these papers trying to solve?",
        "examples": [
            "What problem are all these papers solving?",
            "How are their approaches different?",
            "What limitations do the authors mention?",
        ],
    },
    "💻 Documentation": {
        "description": "Ask questions across technical documentation and implementation guides.",
        "question": "Explain the main concepts described in these documents.",
        "examples": [
            "How does this technology work?",
            "How would I implement the approach described here?",
            "Compare how these technologies solve the same problem.",
        ],
    },
    "🏢 Company Research": {
        "description": "Research companies using official pages, engineering blogs, careers and reports.",
        "question": "What are the major themes across these company sources?",
        "examples": [
            "What technologies and engineering areas are mentioned?",
            "What are the company's major products?",
            "Generate interview questions from these sources.",
        ],
    },
}


def hostname(url):
    return urlparse(url).netloc.replace("www.", "")
