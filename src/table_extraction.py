
"""Structured extraction of digitally detectable PDF tables."""

import hashlib
import re
from pathlib import Path

import fitz


def _document_prefix(document, document_id=None):
    """Generate a deterministic, collision-resistant document prefix."""

    if document_id is not None:
        raw_id = str(document_id)

        slug = re.sub(
            r"[^a-zA-Z0-9_-]+", "_", raw_id
        ).strip("_-").lower()

        if not slug:
            raise ValueError("Invalid document_id.")

        digest = hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()[:12]

        return f"{slug}_{digest}"

    stem = Path(document).stem.lower()

    slug = re.sub(
        r"[^a-z0-9]+", "_", stem
    ).strip("_")

    if not slug:
        raise ValueError("Invalid document filename.")

    return slug


def _reconstruct_cell(words, bbox):
    """Reconstruct cell text from positioned PDF words."""

    if bbox is None:
        return ""

    rect = fitz.Rect(bbox)
    selected = []

    for word in words:
        x0, y0, x1, y1, value = word[:5]

        center = fitz.Point(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
        )

        if rect.contains(center):
            selected.append(word)

    lines = {}

    for word in selected:
        key = (word[5], word[6])
        lines.setdefault(key, []).append(word)

    ordered_lines = sorted(
        lines.values(),
        key=lambda group: (
            min(w[1] for w in group),
            min(w[0] for w in group),
        ),
    )

    return '\n'.join(
        " ".join(
            str(word[4])
            for word in sorted(
                group,
                key=lambda w: (w[7], w[0]),
            )
        )
        for group in ordered_lines
    ).strip()



def _normalize_table_columns(table, words):
    """
    Conservatively normalize narrow, empty column artifacts.

    The original cells and coordinates are retained for
    auditing. This heuristic does not guarantee correctness
    for every PDF layout.
    """

    original_cells = [
        [
            _reconstruct_cell(words, bbox)
            for bbox in row.cells
        ]
        for row in table.rows
    ]

    original_bboxes = [
        [
            list(bbox) if bbox is not None else None
            for bbox in row.cells
        ]
        for row in table.rows
    ]

    widths = []

    for col in range(table.col_count):
        values = [
            row.cells[col][2] - row.cells[col][0]
            for row in table.rows
            if row.cells[col] is not None
        ]

        widths.append(min(values) if values else None)

    removed = []

    for col in range(1, table.col_count - 1):

        width = widths[col]

        if width is None or width > 3:
            continue

        if any(row[col].strip() for row in original_cells):
            continue

        left = widths[col - 1]
        right = widths[col + 1]

        if (
            left is not None
            and right is not None
            and left >= 20
            and right >= 20
        ):
            removed.append(col)

    kept = [
        col for col in range(table.col_count)
        if col not in removed
    ]

    normalized_cells = [
        [row[col] for col in kept]
        for row in original_cells
    ]

    normalized_bboxes = [
        [row[col] for col in kept]
        for row in original_bboxes
    ]

    return {
        "cells": normalized_cells,
        "cell_bboxes": normalized_bboxes,
        "original_cells": original_cells,
        "original_cell_bboxes": original_bboxes,
        "original_column_count": table.col_count,
        "removed_artifact_columns": removed,
        "column_widths": widths,
        "normalization_applied": bool(removed),
        "normalization_method": "narrow_empty_border_heuristic",
    }



def _recover_physical_grid(page, table, words):
    """
    Recover a rectangular physical grid conservatively.

    Grid boundaries are derived from connected PDF
    drawing geometry, not arbitrary bbox extensions.

    Acceptance requires:
      - complete horizontal and vertical grid lines
      - no suspicious large intra-row text gaps
      - no loss of original detected table words

    This remains a heuristic, not a correctness proof.
    """

    bbox = fitz.Rect(table.bbox)
    tolerance = 1.5

    horizontal = []
    vertical = []

    for drawing in page.get_drawings():
        for item in drawing["items"]:

            if item[0] != "l":
                continue

            p1, p2 = item[1], item[2]

            if abs(p1.y - p2.y) <= tolerance:
                horizontal.append((
                    min(p1.x, p2.x),
                    max(p1.x, p2.x),
                    (p1.y + p2.y) / 2,
                ))

            elif abs(p1.x - p2.x) <= tolerance:
                vertical.append((
                    (p1.x + p2.x) / 2,
                    min(p1.y, p2.y),
                    max(p1.y, p2.y),
                ))

    def review(reason, **details):
        return {
            "status": "needs_review",
            "reason": reason,
            **details,
        }

    def unique(values):
        result = []

        for value in sorted(values):
            if (
                not result
                or abs(value - result[-1]) > tolerance
            ):
                result.append(float(value))

        return result

    # Find horizontal lines overlapping the detected
    # table's actual vertical extent.
    candidates = [
        line for line in horizontal
        if (
            line[1] >= bbox.x0 - tolerance
            and line[0] <= bbox.x1 + tolerance
            and bbox.y0 - 20 <= line[2] <= bbox.y1 + 50
        )
    ]

    if not candidates:
        return review("missing_horizontal_boundaries")

    # Use the widest overlapping horizontal line as
    # a candidate physical table width.
    anchor = max(
        candidates,
        key=lambda line: line[1] - line[0]
    )

    left, right, _ = anchor

    # Retain only horizontal lines spanning the
    # same physical table width.
    horizontal = [
        line for line in candidates
        if (
            abs(line[0] - left) <= tolerance
            and abs(line[1] - right) <= tolerance
        )
    ]

    ys = unique([line[2] for line in horizontal])

    if len(ys) < 3:
        return review("insufficient_horizontal_boundaries")

    top = ys[0]
    bottom = ys[-1]

    # Vertical boundaries must connect the recovered
    # horizontal grid's top and bottom.
    vertical = [
        line for line in vertical
        if (
            left - tolerance <= line[0] <= right + tolerance
            and line[1] <= top + tolerance
            and line[2] >= bottom - tolerance
        )
    ]

    xs = unique([line[0] for line in vertical])

    if len(xs) < 3:
        return review("insufficient_vertical_boundaries")

    if (
        abs(xs[0] - left) > tolerance
        or abs(xs[-1] - right) > tolerance
    ):
        return review("incomplete_outer_vertical_boundaries")

    # Ensure all horizontal boundaries span the grid.
    for y in ys:
        if not any(
            x0 <= xs[0] + tolerance
            and x1 >= xs[-1] - tolerance
            and abs(line_y - y) <= tolerance
            for x0, x1, line_y in horizontal
        ):
            return review("incomplete_horizontal_grid")

    # Ensure all vertical boundaries span the grid.
    for x in xs:
        if not any(
            abs(line_x - x) <= tolerance
            and y0 <= ys[0] + tolerance
            and y1 >= ys[-1] - tolerance
            for line_x, y0, y1 in vertical
        ):
            return review("incomplete_vertical_grid")

    cells = []
    cell_bboxes = []
    suspicious_rows = []

    recovered_word_indices = set()

    for r in range(len(ys) - 1):
        row = []
        bbox_row = []

        for c in range(len(xs) - 1):
            cell = [
                xs[c],
                ys[r],
                xs[c + 1],
                ys[r + 1],
            ]

            rect = fitz.Rect(cell)

            selected = [
                (index, word)
                for index, word in enumerate(words)
                if rect.contains(
                    fitz.Point(
                        (word[0] + word[2]) / 2,
                        (word[1] + word[3]) / 2,
                    )
                )
            ]

            recovered_word_indices.update(
                index for index, _ in selected
            )

            row.append(
                _reconstruct_cell(words, cell)
            )

            bbox_row.append(cell)

            groups = {}

            for _, word in selected:
                groups.setdefault(
                    (word[5], word[6]), []
                ).append(word)

            ranges = sorted(
                (
                    min(w[1] for w in group),
                    max(w[3] for w in group),
                )
                for group in groups.values()
            )

            for previous, current in zip(
                ranges, ranges[1:]
            ):
                gap = current[0] - previous[1]

                if gap > 15:
                    suspicious_rows.append({
                        "row": r,
                        "column": c,
                        "gap": round(gap, 2),
                    })

        cells.append(row)
        cell_bboxes.append(bbox_row)

    if suspicious_rows:
        return review(
            "suspicious_intra_row_text_gaps",
            suspicious_rows=suspicious_rows,
        )

    # Audit the original detected cell regions.
    original_word_indices = set()

    for original_row in table.rows:
        for original_bbox in original_row.cells:
            if original_bbox is None:
                continue

            rect = fitz.Rect(original_bbox)

            for index, word in enumerate(words):
                center = fitz.Point(
                    (word[0] + word[2]) / 2,
                    (word[1] + word[3]) / 2,
                )

                if rect.contains(center):
                    original_word_indices.add(index)

    missing_indices = (
        original_word_indices - recovered_word_indices
    )

    if missing_indices:
        return review(
            "original_table_words_not_recovered",
            missing_words=[
                words[index][4]
                for index in sorted(missing_indices)
            ],
        )

    return {
        "status": "accepted",
        "reason": "complete_grid_no_large_text_gaps",
        "cells": cells,
        "cell_bboxes": cell_bboxes,
        "row_count": len(cells),
        "column_count": len(cells[0]),
        "recovery_method": "complete_grid_with_gap_check",
        "horizontal_boundaries": ys,
        "vertical_boundaries": xs,
        "word_coverage_verified": True,
    }


def extract_pdf_tables(
    pdf_path,
    document_id=None,
    document=None,
):
    """
    Extract structured tables from a digital PDF.

    Scanned tables are not supported by this function.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    filename = document or pdf_path.name

    prefix = _document_prefix(
        filename,
        document_id=document_id,
    )

    extracted_tables = []

    pdf = fitz.open(str(pdf_path))

    try:
        for page_number, page in enumerate(
            pdf, start=1
        ):
            finder = page.find_tables()

            words = page.get_text(
                "words",
                sort=True,
            )

            for table_number, table in enumerate(
                finder.tables, start=1
            ):
                normalized = _normalize_table_columns(
                    table, words
                )

                recovery = _recover_physical_grid(
                    page, table, words
                )

                if recovery["status"] == "accepted":
                    reconstructed_rows = recovery["cells"]
                    reconstructed_bboxes = recovery[
                        "cell_bboxes"
                    ]
                    row_count = recovery["row_count"]
                    column_count = recovery["column_count"]
                    extraction_method = (
                        "complete_grid_with_gap_check"
                    )
                else:
                    reconstructed_rows = normalized["cells"]
                    reconstructed_bboxes = normalized[
                        "cell_bboxes"
                    ]
                    row_count = len(reconstructed_rows)
                    column_count = (
                        len(reconstructed_rows[0])
                        if reconstructed_rows else 0
                    )
                    extraction_method = "pymupdf_words"

                if not reconstructed_rows:
                    continue

                headers = reconstructed_rows[0]
                data_rows = reconstructed_rows[1:]

                table_id = (
                    f"{prefix}_page_{page_number}"
                    f"_table_{table_number:03d}"
                )

                record = {
                    "table_id": table_id,
                    "document": filename,
                    "page": page_number,
                    "page_numbers": [page_number],
                    "table_number": table_number,
                    "bbox": list(table.bbox),
                    "row_count": row_count,
                    "column_count": column_count,
                    "cell_bboxes": reconstructed_bboxes,
                    "original_cells": normalized["original_cells"],
                    "original_cell_bboxes": normalized[
                        "original_cell_bboxes"
                    ],
                    "original_column_count": normalized[
                        "original_column_count"
                    ],
                    "removed_artifact_columns": normalized[
                        "removed_artifact_columns"
                    ],
                    "column_widths": normalized["column_widths"],
                    "normalization_applied": normalized[
                        "normalization_applied"
                    ],
                    "normalization_method": normalized[
                        "normalization_method"
                    ],
                    "headers": headers,
                    "rows": data_rows,
                    "cells": reconstructed_rows,
                    "extraction_method": extraction_method,
                    "recovery_status": recovery["status"],
                    "recovery_reason": recovery["reason"],
                    "requires_review": (
                        recovery["status"] != "accepted"
                    ),
                    "header_status": "assumed_first_row",
                    "column_structure_status": (
                        "unverified"
                    ),
                    "retrieval_eligible": False,
                    "retrieval_exclusion_reason": (
                        "table_structure_not_verified"
                    ),
                    "suspicious_rows": recovery.get(
                        "suspicious_rows", []
                    ),
                    "raw_detected_row_count": table.row_count,
                    "raw_detected_column_count": table.col_count,
                }

                if document_id is not None:
                    record["document_id"] = str(
                        document_id
                    )

                extracted_tables.append(record)

    finally:
        pdf.close()

    return extracted_tables


def validate_extracted_tables(tables):
    """Validate structural consistency and table identity."""

    errors = []
    seen_ids = set()

    for table in tables:
        table_id = table["table_id"]

        if table_id in seen_ids:
            errors.append(
                f"Duplicate table ID: {table_id}"
            )

        seen_ids.add(table_id)

        cells = table["cells"]
        columns = table["column_count"]

        if len(cells) != table["row_count"]:
            errors.append(
                f"Row count mismatch: {table_id}"
            )

        if any(
            len(row) != columns
            for row in cells
        ):
            errors.append(
                f"Column count mismatch: {table_id}"
            )

        if not table["page_numbers"]:
            errors.append(
                f"Missing page reference: {table_id}"
            )

        if (
            "document_id" in table
            and not table["document_id"]
        ):
            errors.append(
                f"Empty document identity: {table_id}"
            )

    return {
        "valid": not errors,
        "total_tables": len(tables),
        "errors": errors,
    }


def is_table_retrieval_eligible(record):
    """
    Fail-closed gate for trusted legal RAG table retrieval.

    This function does not verify table semantics.
    Records must be explicitly approved by a separate
    verification or review process.
    """

    if not isinstance(record, dict):
        return False

    if record.get("retrieval_eligible") is not True:
        return False

    if record.get("requires_review") is not False:
        return False

    if record.get("recovery_status") != "accepted":
        return False

    if record.get("header_status") != "verified":
        return False

    if record.get("column_structure_status") != "verified":
        return False

    if not isinstance(record.get("table_id"), str):
        return False

    if not record["table_id"].strip():
        return False

    pages = record.get("page_numbers")

    if (
        not isinstance(pages, list)
        or not pages
        or any(
            type(page) is not int or page < 1
            for page in pages
        )
    ):
        return False

    cells = record.get("cells")

    if (
        not isinstance(cells, list)
        or not cells
        or not isinstance(record.get("row_count"), int)
        or not isinstance(record.get("column_count"), int)
    ):
        return False

    if record["row_count"] != len(cells):
        return False

    columns = record["column_count"]

    if columns < 1:
        return False

    if any(
        not isinstance(row, list)
        or len(row) != columns
        or any(
            not isinstance(cell, str)
            for cell in row
        )
        for row in cells
    ):
        return False

    if (
        "document_id" in record
        and (
            not isinstance(record["document_id"], str)
            or not record["document_id"].strip()
        )
    ):
        return False

    return True
