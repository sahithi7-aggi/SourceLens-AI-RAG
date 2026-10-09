from pathlib import Path
from src.loader import load_pdfs


class TestUploadedFile:
    def __init__(self, path):
        self.name = Path(path).name
        self._data = Path(path).read_bytes()

    def getvalue(self):
        return self._data


pdf_path = input("Enter the full path to a PDF: ").strip().strip('"')
uploaded_file = TestUploadedFile(pdf_path)

documents, errors, source_results = load_pdfs([uploaded_file], enable_ocr=False)

print("\nDOCUMENTS:", len(documents))
print("ERRORS:", errors)
print("RESULTS:", source_results)

for document in documents[:3]:
    print("\n--- PAGE ---")
    print("Source:", document.metadata.get("source"))
    print("Page:", document.metadata.get("page"))
    print("Method:", document.metadata.get("extraction_method"))
    print(document.page_content[:500])
