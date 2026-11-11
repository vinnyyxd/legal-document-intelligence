
import ast
from pathlib import Path

import pytest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def app_tree():
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def find_calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and call_name(node.func) == name
    ]


def keyword_value(call, name):
    matches = [
        keyword.value
        for keyword in call.keywords
        if keyword.arg == name
    ]
    assert len(matches) == 1
    return matches[0]


def containing_status(tree, target):
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue

        if not any(
            isinstance(item.context_expr, ast.Call)
            and call_name(item.context_expr.func) == "status"
            for item in node.items
        ):
            continue

        if node.lineno <= target.lineno <= node.end_lineno:
            return node

    return None


@pytest.mark.parametrize(
    "function_name",
    [
        "run_document_pipeline",
        "ingest_document",
    ],
)
def test_pipeline_receives_progress_callback(
    app_tree, function_name
):
    calls = find_calls(app_tree, function_name)
    assert len(calls) == 1

    callback = keyword_value(
        calls[0],
        "progress_callback",
    )

    assert isinstance(callback, ast.Name)
    assert callback.id == "progress_callback"


def test_progress_renderer_imported(app_tree):
    imports = [
        node
        for node in app_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.embedding_progress_ui"
    ]

    assert len(imports) == 1
    assert any(
        alias.name == "make_embedding_progress_callback"
        for alias in imports[0].names
    )


def test_both_progress_renderers_created(app_tree):
    assert len(find_calls(
        app_tree,
        "make_embedding_progress_callback",
    )) == 2

    assert len(find_calls(app_tree, "progress")) >= 2
    assert len(find_calls(app_tree, "empty")) >= 2


@pytest.mark.parametrize(
    "function_name",
    [
        "run_document_pipeline",
        "ingest_document",
    ],
)
def test_progress_created_inside_status_block(
    app_tree, function_name
):
    pipeline_calls = find_calls(
        app_tree,
        function_name,
    )
    assert len(pipeline_calls) == 1

    status_block = containing_status(
        app_tree,
        pipeline_calls[0],
    )

    assert status_block is not None

    assert len(find_calls(
        status_block,
        "make_embedding_progress_callback",
    )) == 1

    assert len(find_calls(
        status_block,
        "progress",
    )) >= 1

    assert len(find_calls(
        status_block,
        "empty",
    )) >= 1


def test_upload_preserves_case_association(app_tree):
    upload = find_calls(
        app_tree,
        "ingest_document",
    )[0]

    case_id = keyword_value(upload, "case_id")

    assert isinstance(case_id, ast.Name)
    assert case_id.id == "active_case_id"


def test_recovery_preserves_eligibility_checks(app_tree):
    recovery = find_calls(
        app_tree,
        "run_document_pipeline",
    )[0]

    enclosing = [
        node
        for node in ast.walk(app_tree)
        if isinstance(node, ast.Try)
        and node.lineno < recovery.lineno < node.end_lineno
    ]

    assert enclosing

    source = ast.unparse(enclosing[-1])

    assert "get_case(active_case_id)" in source
    assert "get_case_documents(active_case_id)" in source
    assert '"chunked"' in source or "'chunked'" in source
