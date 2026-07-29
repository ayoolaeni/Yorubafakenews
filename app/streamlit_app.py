"""Run with: make app  (i.e. streamlit run app/streamlit_app.py)

Interactive demo: paste Yoruba text, get a GENUINE/FAKE call from whichever
model src.evaluation.report picked as best.
"""
from __future__ import annotations

import streamlit as st

from src.models.predictor import load_best_predictor

st.set_page_config(page_title="Yoruba Fake News Detector", page_icon="📰")
st.title("Yoruba Fake News Detector")


@st.cache_resource
def get_predictor():
    return load_best_predictor()


try:
    predictor = get_predictor()
except FileNotFoundError:
    st.error(
        "No trained model available yet. Run `make corpus`, `make preprocess`, "
        "`make train`, and `make evaluate` first."
    )
    st.stop()

st.caption(f"Serving **{predictor.model_name}** trained on the **{predictor.variant}** variant")

text = st.text_area("Paste Yoruba news text to check", height=200)
if st.button("Check", type="primary") and text.strip():
    result = predictor.predict_one(text)
    if result["label"] == "genuine":
        st.success(f"GENUINE — confidence {result['confidence']:.1%}")
    else:
        st.error(f"FAKE — confidence {result['confidence']:.1%}")
    with st.expander("Preprocessed text seen by the model"):
        st.write(result["processed_text"] or "*(empty after preprocessing)*")