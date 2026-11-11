
"""Render document embedding progress inside Streamlit."""

def make_embedding_progress_callback(progress_bar, message):
    """
    Return a callback for document embedding progress events.

    progress_bar:
        A Streamlit st.progress() object.

    message:
        A Streamlit st.empty() placeholder.
    """

    def callback(event):
        if not isinstance(event, dict):
            raise TypeError("Progress event must be a dictionary.")

        event_name = event.get("event")

        completed = event.get("completed", 0)
        total = event.get("total", 0)
        reused = event.get("reused", 0)
        generated = event.get("generated", 0)

        if not isinstance(completed, int):
            raise TypeError("Completed count must be an integer.")

        if not isinstance(total, int):
            raise TypeError("Total count must be an integer.")

        if total < 0 or completed < 0 or completed > total:
            raise ValueError("Invalid embedding progress counts.")

        fraction = completed / total if total else 0.0

        progress_bar.progress(
            fraction,
            text=f"Embeddings: {completed}/{total}",
        )

        if event_name == "checkpoint_loaded":
            message.info(
                f"Recovered {reused} saved embeddings. "
                f"{total - completed} remaining."
            )

        elif event_name == "batch_started":
            message.info(
                f"Generating embeddings... "
                f"{completed}/{total} completed."
            )

        elif event_name == "retry_wait":
            wait = float(event.get("wait_seconds", 0))
            attempt = event.get("attempt", "?")
            maximum = event.get("max_attempts", "?")

            message.warning(
                "Temporary Gemini API error. "
                f"Retry {attempt}/{maximum}; "
                f"waiting {wait:.0f} seconds."
            )

        elif event_name == "batch_saved":
            message.info(
                f"Saved embedding checkpoint: "
                f"{completed}/{total} completed "
                f"({generated} newly generated)."
            )

        elif event_name == "embedding_complete":
            message.success(
                f"Embedding complete: {total}/{total} "
                f"({reused} reused, {generated} generated)."
            )

        else:
            raise ValueError(
                f"Unknown embedding progress event: {event_name}"
            )

    return callback
