"""Live inference panel — calls the FastAPI serving endpoint."""
from __future__ import annotations

import streamlit as st


def render_live_predict_panel(
    api_base_url: str = "http://localhost:8000/api/v1",
) -> None:
    """
    Streamlit form: text input, approach dropdown, submit button.
    On submit: POST to /classify, show intent, confidence bar, latency badge.
    Use httpx (not requests).
    If API unreachable: st.warning with 'make serve' command — do not crash.
    """
    import httpx

    st.subheader("Live Prediction")
    st.caption("Requires FastAPI server — run `make serve` in a separate terminal")

    with st.form("predict_form"):
        text_input = st.text_area(
            "Customer message",
            placeholder="e.g. I want to cancel my order",
            height=100,
        )
        approach = st.selectbox(
            "Approach",
            options=["fine_tuned", "zero_shot", "few_shot"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        submitted = st.form_submit_button("Classify")

    if not submitted or not text_input.strip():
        return

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{api_base_url}/classify",
                json={"text": text_input.strip(), "approach": approach},
            )
        resp.raise_for_status()
        data = resp.json()

        st.success(f"**Intent:** `{data['intent']}`")
        st.progress(data["confidence"], text=f"Confidence: {data['confidence']:.0%}")
        st.metric("Latency", f"{data['latency_ms']:.1f} ms")

    except httpx.ConnectError:
        st.warning(
            "Cannot reach the API server. "
            "Start it with: `make serve`"
        )
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
