import fitz
import os


def extract_pdf(
    pdf_path,
    document_id=None,
    enable_ocr=False,
    ocr_language="eng",
):
    """Extract PDF pages with optional OCR for scanned pages."""

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    extracted_pages = []

    try:
        for page_num, page in enumerate(doc, start=1):

            text = page.get_text("text", sort=True)
            is_scanned = len(text.strip()) < 30

            ocr_status = "not_attempted"
            ocr_error = None

            if enable_ocr and is_scanned:
                try:
                    textpage = page.get_textpage_ocr(
                        language=ocr_language,
                        dpi=200,
                        full=True,
                    )

                    ocr_text = page.get_text(
                        "text",
                        sort=True,
                        textpage=textpage,
                    )

                    if ocr_text.strip():
                        text = ocr_text
                        ocr_status = "success"
                    else:
                        ocr_status = "no_text"

                except Exception as exc:
                    ocr_status = "failed"
                    ocr_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

            page_data = {
                "document": os.path.basename(pdf_path),
                "page": page_num,
                "text": text,
                "is_scanned": is_scanned,
                "word_count": len(text.split()),
            }

            if document_id is not None:
                page_data["document_id"] = str(document_id)

            if enable_ocr:
                page_data["ocr_status"] = ocr_status

                if ocr_error is not None:
                    page_data["ocr_error"] = ocr_error

            extracted_pages.append(page_data)

    finally:
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
