import streamlit as st
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Fake Review Detector",
    layout="centered"
)

@st.cache_resource
def load_models():
    graph_model = joblib.load("graph_model.pkl")

    tokenizer = AutoTokenizer.from_pretrained(
        "bert_fake_review_model"
    )

    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "bert_fake_review_model"
    )

    bert_model.eval()

    return graph_model, tokenizer, bert_model


graph_model, tokenizer, bert_model = load_models()

st.title("Fake Review Detector")

review_text = st.text_area(
    "Enter Review Text",
    height=150
)

rating = st.slider(
    "Rating",
    min_value=1,
    max_value=5,
    value=5
)

st.subheader("Analysis Mode")

advanced_mode = st.checkbox(
    "Enable Advanced Reviewer Analysis",
    value=False
)

if advanced_mode:

    st.subheader("Reviewer Behavior")

    user_review_count = st.number_input(
        "Reviewer Review Count",
        min_value=0,
        value=1
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=1,
        value=30
    )

    user_business_count = st.number_input(
        "Unique Restaurants Reviewed",
        min_value=0,
        value=1
    )

    user_rating_std = st.number_input(
        "Rating Standard Deviation",
        min_value=0.0,
        value=0.0
    )

    reviewer_cluster = st.number_input(
        "Reviewer Cluster Score",
        min_value=0.0,
        value=0.0
    )

    co_review_degree = st.number_input(
        "Co-review Degree",
        min_value=0,
        value=0
    )

    co_review_overlap = st.number_input(
        "Co-review Overlap",
        min_value=0.0,
        max_value=1.0,
        value=0.0
    )

    st.subheader("Temporal Signal")

    burst_score = st.slider(
        "Burst Score",
        0.0,
        1.0,
        0.2
    )

else:
  
    user_review_count = 10
    account_age_days = 365
    user_business_count = 10
    user_rating_std = 1.0
    reviewer_cluster = 0.0
    co_review_degree = 0
    co_review_overlap = 0.0
    burst_score = 0.5

def get_bert_score(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )

    inputs.pop("token_type_ids", None)

    with torch.no_grad():

        logits = bert_model(**inputs).logits

        probs = torch.softmax(
            logits,
            dim=1
        )

    return float(probs[0, 1])


if st.button("Detect Fake Review"):

    if not review_text.strip():

        st.warning(
            "Please enter a review."
        )

    else:

        # BERT score
        text_score = get_bert_score(
            review_text
        )

        # Graph features

        reviews_per_day = (
            user_review_count /
            max(account_age_days, 1)
        )

        graph_features = pd.DataFrame([{

            "user_review_count":
                user_review_count,

            "account_age_days":
                account_age_days,

            "reviews_per_day":
                reviews_per_day,

            "user_business_count":
                user_business_count,

            "user_rating_std":
                user_rating_std,

            "reviewer_cluster":
                reviewer_cluster,

            "co_review_degree":
                co_review_degree,

            "co_review_overlap":
                co_review_overlap

        }])

        graph_score = float(
            graph_model.predict_proba(
                graph_features
            )[0, 1]
        )

        # Fusion score

        final_score = (
            0.5 * text_score +
            0.3 * graph_score +
            0.2 * burst_score
        )

        fraud_probability = round(
            final_score * 100,
            1
        )

        st.subheader("Prediction")

        if final_score >= 0.5:

            st.error(
                f"🚨 Likely Fake Review ({fraud_probability}% confidence)"
            )

        else:

            st.success(
                f"Likely Genuine Review ({100 - fraud_probability}% confidence)"
            )

        st.progress(
            int(final_score * 100)
        )

        st.metric(
            "Fraud Probability",
            f"{fraud_probability}%"
        )

        with st.expander(
            "View Detailed Analysis"
        ):

            st.write(
                f"BERT Text Score: {text_score:.3f}"
            )

            st.write(
                f"Graph Behavior Score: {graph_score:.3f}"
            )

            st.write(
                f"Burst Score: {burst_score:.3f}"
            )

            st.write(
                f"Final Fusion Score: {final_score:.3f}"
            )
