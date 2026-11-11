"""
Access control for the private Streamlit portfolio demo.

This is a shared-password gate, not per-user authentication.
"""

import hmac
import secrets
import streamlit as st


def password_matches(provided, expected):
    """Compare passwords without ordinary string equality."""
    if not isinstance(provided, str):
        return False

    if not isinstance(expected, str) or not expected:
        return False

    return hmac.compare_digest(
        provided.encode("utf-8"),
        expected.encode("utf-8"),
    )


_server_hmac_key = secrets.token_bytes(32)
_active_password_mac = None
_active_token = None

def require_demo_access():
    """Stop rendering until the configured demo password is supplied."""

    try:
        expected = st.secrets["DEMO_PASSWORD"]
    except (KeyError, FileNotFoundError):
        st.error("Demo access is not configured.")
        st.stop()

    if not isinstance(expected, str) or not expected:
        st.error("Demo access is not configured.")
        st.stop()

    # Rotate the server-side token when the configured password changes.
    # No password-derived value is stored in session state.
    global _active_password_mac, _active_token

    password_mac = hmac.digest(
        _server_hmac_key,
        expected.encode("utf-8"),
        "sha256",
    )

    if (
        _active_password_mac is None
        or not hmac.compare_digest(
            _active_password_mac,
            password_mac,
        )
    ):
        _active_password_mac = password_mac
        _active_token = secrets.token_urlsafe(32)

    if (
        st.session_state.get("demo_authenticated") is True
        and st.session_state.get("demo_auth_token")
        == _active_token
    ):
        return

    st.session_state["demo_authenticated"] = False
    st.session_state.pop("demo_auth_token", None)

    st.title("Legal Document Intelligence")
    st.caption("Private portfolio demonstration")

    with st.form("demo_login_form"):
        provided = st.text_input(
            "Demo password",
            type="password",
        )
        submitted = st.form_submit_button("Enter Demo")

    if submitted:
        if password_matches(provided, expected):
            st.session_state["demo_authenticated"] = True
            st.session_state["demo_auth_token"] = _active_token
            st.rerun()
        else:
            st.error("Incorrect demo password.")

    st.stop()
