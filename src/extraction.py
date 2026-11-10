
import fitz
import os

def extract_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    extracted_pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True)
        is_scanned = len(text.strip()) < 30

        page_data = {
            "document": os.path.basename(pdf_path),
            "page": page_num,
            "text": text,
            "is_scanned": is_scanned,
            "word_count": len(text.split())
        }

        extracted_pages.append(page_data)

    doc.close()
    return extracted_pages


def print_extraction_summary(pages):
    total_pages = len(pages)
    scanned_pages = sum(1 for page in pages if page["is_scanned"])
    total_words = sum(page["word_count"] for page in pages)

    print("PDF EXTRACTION SUMMARY")
    print("----------------------")
    print(f"Total pages: {total_pages}")
    print(f"Total words: {total_words}")
    print(f"Potentially scanned pages: {scanned_pages}")
    print(f"Digitally readable pages: {total_pages - scanned_pages}")
