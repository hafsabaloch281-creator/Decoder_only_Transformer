import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="Ahm-d18/my-bert-sentiment-model")

st.title("Multilingual Sentiment Analysis")
clf = load_model()

text = st.text_area("Enter some text")

if st.button("Analyze") and text.strip():
    result = clf(text)[0]
    st.subheader(result["label"])
    st.progress(float(result["score"]))
    st.write(f"Confidence: {result['score']:.2%}")