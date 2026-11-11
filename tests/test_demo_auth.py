from src.demo_auth import password_matches


def test_correct_password():
    assert password_matches("demo-123", "demo-123")


def test_incorrect_password():
    assert not password_matches("wrong", "demo-123")


def test_empty_password_rejected():
    assert not password_matches("", "")


def test_invalid_password_types_rejected():
    assert not password_matches(None, "demo-123")
    assert not password_matches("demo-123", None)


def test_unicode_password():
    assert password_matches("sicher-🔐", "sicher-🔐")


# --------------------------------------------------
# Streamlit authentication behavior
# --------------------------------------------------

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class StopExecution(Exception):
    pass


class RerunExecution(Exception):
    pass


class FakeStreamlit:
    def __init__(
        self,
        password="demo-secret",
        submitted=False,
        provided="",
        session=None,
    ):
        self.secrets = (
            {} if password is None
            else {"DEMO_PASSWORD": password}
        )
        self.session_state = (
            {} if session is None else session
        )
        self.submitted = submitted
        self.provided = provided
        self.errors = []

    def error(self, message):
        self.errors.append(message)

    def stop(self):
        raise StopExecution()

    def rerun(self):
        raise RerunExecution()

    def title(self, message):
        pass

    def caption(self, message):
        pass

    @contextmanager
    def form(self, key):
        yield

    def text_input(self, label, type=None):
        return self.provided

    def form_submit_button(self, label):
        return self.submitted


def run_auth(fake):
    import src.demo_auth as demo_auth

    with patch.object(demo_auth, "st", fake):
        return demo_auth.require_demo_access()


def test_missing_secret_blocks_access():
    fake = FakeStreamlit(password=None)

    with pytest.raises(StopExecution):
        run_auth(fake)

    assert fake.errors


def test_wrong_password_blocks_access():
    fake = FakeStreamlit(
        submitted=True,
        provided="incorrect",
    )

    with pytest.raises(StopExecution):
        run_auth(fake)

    assert fake.session_state.get(
        "demo_authenticated"
    ) is not True


def test_correct_password_authenticates():
    fake = FakeStreamlit(
        submitted=True,
        provided="demo-secret",
    )

    with pytest.raises(RerunExecution):
        run_auth(fake)

    assert fake.session_state["demo_authenticated"] is True
    assert fake.session_state["demo_auth_token"]


def test_authenticated_session_can_continue():
    fake = FakeStreamlit(
        submitted=True,
        provided="demo-secret",
    )

    with pytest.raises(RerunExecution):
        run_auth(fake)

    fake.submitted = False
    fake.provided = ""

    assert run_auth(fake) is None


def test_password_rotation_invalidates_session():
    fake = FakeStreamlit(
        submitted=True,
        provided="demo-secret",
    )

    with pytest.raises(RerunExecution):
        run_auth(fake)

    fake.secrets["DEMO_PASSWORD"] = "new-secret"
    fake.submitted = False
    fake.provided = ""

    with pytest.raises(StopExecution):
        run_auth(fake)

    assert fake.session_state["demo_authenticated"] is False


def test_removed_secret_invalidates_access():
    fake = FakeStreamlit(
        submitted=True,
        provided="demo-secret",
    )

    with pytest.raises(RerunExecution):
        run_auth(fake)

    fake.secrets.clear()

    with pytest.raises(StopExecution):
        run_auth(fake)



def test_session_contains_no_password_fingerprint():
    fake = FakeStreamlit(
        submitted=True,
        provided="demo-secret",
    )

    with pytest.raises(RerunExecution):
        run_auth(fake)

    assert "demo_password_fingerprint" not in fake.session_state
    assert "demo_auth_token" in fake.session_state
    assert fake.session_state["demo_auth_token"] != "demo-secret"
