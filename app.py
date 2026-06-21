import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ABSA — Aspect-Based Sentiment Analysis",
    page_icon="🔍",
    layout="centered"
)

st.title("Aspect-Based Sentiment Analysis")
st.markdown("Analyse sentiment for specific aspects in e-commerce reviews using fine-tuned RoBERTa.")

st.divider()

with st.form("predict_form"):
    text = st.text_area(
        "Review text",
        placeholder="e.g. The battery life on this laptop is amazing but the keyboard feels cheap.",
        height=120
    )
    aspect = st.text_input(
        "Aspect term",
        placeholder="e.g. battery life"
    )
    submitted = st.form_submit_button("Analyse", use_container_width=True)

if submitted:
    if not text.strip() or not aspect.strip():
        st.error("Please enter both a review and an aspect term.")
    else:
        with st.spinner("Analysing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text, "aspect": aspect}
                )
                data = response.json()["prediction"]

                sentiment = data["predicted_sentiment"]
                confidence = data["confidence"]
                probs = data["probabilities"]

                color_map = {
                    "positive": "🟢",
                    "negative": "🔴",
                    "neutral": "🟡"
                }

                st.divider()
                st.subheader("Result")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Predicted sentiment",
                        value=f"{color_map[sentiment]} {sentiment.capitalize()}"
                    )
                with col2:
                    st.metric(
                        label="Confidence",
                        value=f"{confidence * 100:.1f}%"
                    )

                st.subheader("Probability breakdown")
                labels = list(probs.keys())
                values = list(probs.values())
                colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.barh(labels, values, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                for bar, val in zip(bars, values):
                    ax.text(
                        val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=11
                    )
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.divider()
                st.subheader("Input sent to model")
                st.code(f"[ASPECT] {aspect} [SEP] {text}")

            except Exception as e:
                st.error(f"API error: {e}")

st.divider()
st.caption("Fine-tuned RoBERTa · SemEval 2014 · MLflow + Optuna · SHAP XAI")