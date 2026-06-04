import streamlit as st
import torch
from src.predict import EmotionPredictor

# ---- Page config ----
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🎭",
    layout="centered"
)

# ---- Load model (cached so it only loads once) ----
@st.cache_resource
def load_model():
    return EmotionPredictor(model_path="models/best_distilbert.pt")c

predictor = load_model()

# ---- UI ----
st.title("🎭 Emotion Classifier")
st.markdown("Detects emotions in text using a fine-tuned DistilBERT model trained on Google's GoEmotions dataset (28 emotion categories).")

st.divider()

# Input
text_input = st.text_area(
    "Enter text to analyze:",
    placeholder="e.g. I'm so proud of what you've accomplished!",
    height=100
)

# Analyze button
if st.button("Analyze Emotion", type="primary", use_container_width=True):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            result = predictor.predict(text_input)
        
        # Main prediction
        st.markdown("### Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Emotion", result["predicted_emotion"].title())
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        # Top 3 predictions as a bar chart
        st.markdown("### Top 3 Predictions")
        chart_data = {
            item["emotion"].title(): item["confidence"]
            for item in result["top_3"]
        }
        st.bar_chart(chart_data, horizontal=True)
        
    else:
        st.warning("Please enter some text.")

# ---- Sidebar with project info ----
with st.sidebar:
    st.markdown("### About This Project")
    st.markdown(
        """
        **Model:** DistilBERT (fine-tuned)  
        **Dataset:** GoEmotions (58K Reddit comments)  
        **Emotions:** 28 categories  
        **Baseline Accuracy:** 41.95%  
        **Model Accuracy:** 58.12%  
        """
    )
    
    st.divider()
    
    st.markdown("### Try These Examples")
    examples = [
        "I'm so happy for you, congratulations!",
        "This is absolutely disgusting behavior",
        "I'm nervous about the interview tomorrow",
        "Thank you for helping me, you're amazing",
        "I can't believe they cancelled the show",
        "Haha that's the funniest thing I've seen"
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["example_text"] = ex
            st.rerun()

# Handle example button clicks
if "example_text" in st.session_state:
    text_input = st.session_state.pop("example_text")
    st.rerun()