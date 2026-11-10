
import re
from pathlib import Path


# --------------------------------------------------
# 1. Legal heading patterns
# --------------------------------------------------

CHAPTER_PATTERN = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\s*$",
    re.IGNORECASE
)

SECTION_PATTERN = re.compile(
    r"^\s*SECTION\s+([IVXLCDM]+|\d+)\s*$",
    re.IGNORECASE
)

ARTICLE_PATTERN = re.compile(
    r"^\s*Article\s+(\d+[a-z]?)\s*$",
    re.IGNORECASE
)

PARAGRAPH_PATTERN = re.compile(
    r"^\s*(\d{1,3})\.\s+(.+?)\s*$"
)

SUBCLAUSE_PATTERN = re.compile(
    r"^\s*\(([a-z])\)\s*(.*)$",
    re.IGNORECASE
)


# --------------------------------------------------
# 2. Detect legal structure
# --------------------------------------------------

def detect_legal_structure(pages):

    structure = []

    for page in pages:

        page_number = page["page"]

        for line in page["text"].splitlines():

            line = line.strip()

            chapter = CHAPTER_PATTERN.fullmatch(line)
            section = SECTION_PATTERN.fullmatch(line)
            article = ARTICLE_PATTERN.fullmatch(line)

            if chapter:
                structure.append({
                    "type": "chapter",
                    "number": chapter.group(1),
                    "page": page_number
                })

            elif section:
                structure.append({
                    "type": "section",
                    "number": section.group(1),
                    "page": page_number
                })

            elif article:
                structure.append({
                    "type": "article",
                    "number": article.group(1),
                    "page": page_number
                })

    return structure


# --------------------------------------------------
# 3. Split documents into legal paragraphs
# --------------------------------------------------

def split_into_paragraphs(pages):

    paragraphs = []

    current = None

    chapter = None
    chapter_title = None

    section = None
    section_title = None

    article = None
    article_title = None

    pending_title = None

    def flush_current():

        nonlocal current

        if current is None:
            return

        text = " ".join(current["lines"]).strip()

        if text:

            unique_pages = sorted(
                set(current["page_numbers"])
            )

            paragraphs.append({
                "document": current["document"],
                "chapter": current["chapter"],
                "chapter_title": current["chapter_title"],
                "section": current["section"],
                "section_title": current["section_title"],
                "article": current["article"],
                "article_title": current["article_title"],
                "paragraph": current["paragraph"],
                "subclauses": current["subclauses"],
                "start_page": unique_pages[0],
                "end_page": unique_pages[-1],
                "page_numbers": unique_pages,
                "line_records": current["line_records"],
                "text": text
            })

        current = None

    previous_document = None

    for page in pages:

        page_number = page["page"]
        document = page.get("document", "unknown.pdf")

        # Reset the legal hierarchy when a new PDF starts
        if (
            previous_document is not None
            and document != previous_document
        ):

            flush_current()

            chapter = None
            chapter_title = None
            section = None
            section_title = None
            article = None
            article_title = None
            pending_title = None

        previous_document = document

        for raw_line in page["text"].splitlines():

            line = raw_line.strip()

            if not line:
                continue

            # Ignore the known GDPR running page header
            if re.match(
                r"^(?:"
                r"L\s+\d+/\d+"
                r"|\d{1,2}\.\d{4}"
                r"|\d{1,2}\.\d{1,2}\.\d{4}"
                r")\s+EN\s+Official Journal",
                line,
                re.IGNORECASE
            ):
                continue

            # CHAPTER detection
            match = CHAPTER_PATTERN.fullmatch(line)

            if match:

                flush_current()

                chapter = match.group(1)
                chapter_title = None

                section = None
                section_title = None

                pending_title = "chapter"
                continue

            # SECTION detection
            match = SECTION_PATTERN.fullmatch(line)

            if match:

                flush_current()

                section = match.group(1)
                section_title = None

                pending_title = "section"
                continue

            # ARTICLE detection
            match = ARTICLE_PATTERN.fullmatch(line)

            if match:

                flush_current()

                article = match.group(1)
                article_title = None

                pending_title = "article"
                continue

            # Capture the heading title on the following line
            if pending_title:

                if not PARAGRAPH_PATTERN.match(line):

                    if pending_title == "chapter":
                        chapter_title = line

                    elif pending_title == "section":
                        section_title = line

                    elif pending_title == "article":
                        article_title = line

                    pending_title = None
                    continue

                pending_title = None

            # Detect a numbered legal paragraph
            match = PARAGRAPH_PATTERN.match(line)

            if match:

                flush_current()

                paragraph_number = match.group(1)
                paragraph_text = match.group(2).strip()

                current = {
                    "document": document,
                    "chapter": chapter,
                    "chapter_title": chapter_title,
                    "section": section,
                    "section_title": section_title,
                    "article": article,
                    "article_title": article_title,
                    "paragraph": paragraph_number,
                    "lines": [],
                    "page_numbers": [],
                    "line_records": [],
                    "subclauses": []
                }

                if paragraph_text:
                    current["lines"].append(paragraph_text)
                    current["page_numbers"].append(page_number)
                    current["line_records"].append({
                        "page": page_number,
                        "text": paragraph_text
                    })

            # Continuation of the current paragraph
            elif current is not None:

                current["lines"].append(line)
                current["page_numbers"].append(page_number)

                current["line_records"].append({
                    "page": page_number,
                    "text": line
                })

                # Record lettered subclauses
                subclause = SUBCLAUSE_PATTERN.match(line)

                if subclause:

                    letter = subclause.group(1).lower()

                    if letter not in current["subclauses"]:
                        current["subclauses"].append(letter)

    # Save the final paragraph
    flush_current()

    return paragraphs


# --------------------------------------------------
# 4. Create final legal chunks
# --------------------------------------------------


def create_legal_chunks(pages, max_words=800):

    paragraph_records = split_into_paragraphs(pages)

    chunks = []
    chunk_counter = 0

    for record in paragraph_records:

        lines = record.get("line_records", [])

        original_text = record["text"]
        original_word_count = len(original_text.split())

        # Identify actual subclause boundaries.
        # Only markers at the beginning of an extracted line count.

        marker_indices = []

        for i, item in enumerate(lines):

            match = SUBCLAUSE_PATTERN.match(item["text"])

            if match:
                marker_indices.append(
                    (i, match.group(1).lower())
                )

        # Split only unusually long paragraphs
        # that have multiple actual subclauses.

        should_split = (
            original_word_count > max_words
            and len(marker_indices) >= 2
        )

        segments = []

        if should_split:

            current_segment = []
            current_letter = None

            for item in lines:

                match = SUBCLAUSE_PATTERN.match(
                    item["text"]
                )

                if match:

                    # Save the previous segment
                    if current_segment:
                        segments.append((
                            current_letter,
                            current_segment
                        ))

                    # Start a new subclause
                    current_letter = match.group(1).lower()
                    current_segment = [item]

                else:
                    # Add continuation text
                    current_segment.append(item)

            if current_segment:
                segments.append((
                    current_letter,
                    current_segment
                ))

        else:

            # Keep normal-sized paragraphs together
            segments = [(None, lines)]

        # Build individual chunks

        document_stem = Path(
            record["document"]
        ).stem.lower()

        document_slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            document_stem
        ).strip("_")

        article = record["article"]
        paragraph = record["paragraph"]

        article_slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            (article or "unassigned").lower()
        ).strip("_")

        parent_reference = (
            f"Article {article}({paragraph})"
            if article else None
        )

        for subclause, segment_lines in segments:

            if segment_lines:

                text = " ".join(
                    item["text"] for item in segment_lines
                ).strip()

                segment_pages = sorted(set(
                    item["page"] for item in segment_lines
                ))

            else:
                text = original_text
                segment_pages = record["page_numbers"]

            if not text:
                continue

            chunk_counter += 1

            if subclause is not None:

                clause_reference = (
                    f"{parent_reference}({subclause})"
                    if parent_reference else None
                )

                chunk_subclauses = [subclause]

                chunk_suffix = f"_subclause_{subclause}"

            else:

                clause_reference = parent_reference

                chunk_subclauses = (
                    [] if should_split
                    else record["subclauses"]
                )

                chunk_suffix = "_body"

            chunk_id = (
                f"{document_slug}_article_{article_slug}"
                f"_paragraph_{paragraph}"
                f"{chunk_suffix}"
                f"_page_{segment_pages[0]}"
                f"_chunk_{chunk_counter:04d}"
            )

            chunks.append({

                "chunk_id": chunk_id,

                "document": record["document"],

                "chapter": record["chapter"],
                "chapter_title": record["chapter_title"],

                "section": record["section"],
                "section_title": record["section_title"],

                "article": article,
                "article_title": record["article_title"],

                "paragraph": paragraph,

                "subclause": subclause,

                "parent_clause_reference": parent_reference,

                "clause_reference": clause_reference,

                "subclauses": chunk_subclauses,

                "start_page": min(segment_pages),
                "end_page": max(segment_pages),
                "page_numbers": segment_pages,

                "text": text,

                "word_count": len(text.split())
            })

    return chunks



# --------------------------------------------------
# 5. Validate generated chunks
# --------------------------------------------------

def validate_chunks(chunks):

    errors = []
    warnings = []

    seen_ids = set()

    for index, chunk in enumerate(chunks):

        chunk_id = chunk["chunk_id"]

        if chunk_id in seen_ids:
            errors.append(
                f"Duplicate chunk ID: {chunk_id}"
            )

        seen_ids.add(chunk_id)

        if not chunk["text"].strip():
            errors.append(
                f"Empty text in chunk: {chunk_id}"
            )

        if chunk["start_page"] > chunk["end_page"]:
            errors.append(
                f"Invalid page range: {chunk_id}"
            )

        if not chunk["page_numbers"]:
            errors.append(
                f"Missing page numbers: {chunk_id}"
            )

        if chunk["word_count"] > 800:
            warnings.append(
                f"Large paragraph requires review: {chunk_id} "
                f"({chunk['word_count']} words)"
            )

        if chunk["article"] is None:
            warnings.append(
                f"Missing article reference: {chunk_id}"
            )

    return {
        "valid": len(errors) == 0,
        "total_chunks": len(chunks),
        "errors": errors,
        "warnings": warnings
    }
