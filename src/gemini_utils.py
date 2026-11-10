
import re
import time


def generate_with_retry(
    client,
    model,
    contents,
    config,
    fallback_models=None,
    max_attempts=2,
    base_delay=15
):
    """
    Generate Gemini content with bounded retries
    and fallback models.

    Retries temporary API errors only.
    """

    fallback_models = fallback_models or []

    models_to_try = list(dict.fromkeys(
        [model] + fallback_models
    ))

    last_error = None

    for model_name in models_to_try:

        print(f"\nTrying Gemini model: {model_name}")

        for attempt in range(max_attempts):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )

                if not response.text:
                    raise ValueError(
                        f"{model_name} returned an empty response."
                    )

                print(f"Generation successful: {model_name}")

                return response

            except Exception as exc:

                message = str(exc).lower()

                # Identify temporary errors
                status_match = re.search(
                    r"\b(429|500|502|503|504)\b",
                    message
                )

                retryable = (
                    status_match is not None
                    or any(
                        marker in message
                        for marker in [
                            "resource_exhausted",
                            "unavailable",
                            "timeout",
                            "deadline_exceeded",
                            "temporarily"
                        ]
                    )
                )

                if not retryable:
                    raise

                last_error = exc

                # Respect server-provided retry delay
                delay_match = re.search(
                    r"retry(?:delay| in).*?(\d+(?:\.\d+)?)s",
                    message
                )

                if delay_match:
                    wait_time = float(delay_match.group(1)) + 2
                else:
                    wait_time = min(
                        120,
                        base_delay * (2 ** attempt)
                    )

                if attempt < max_attempts - 1:

                    print(
                        f"Temporary API error. "
                        f"Retrying {model_name} in "
                        f"{wait_time:.0f} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    print(
                        f"{model_name} unavailable after "
                        f"{max_attempts} attempts."
                    )

        print("Checking fallback model...")

    raise RuntimeError(
        "All configured Gemini models failed. "
        "Last error: " + str(last_error)
    ) from last_error
