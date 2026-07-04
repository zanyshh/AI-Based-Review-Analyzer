import streamlit as st
import os
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page Config streamlit
# -----------------------------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Train Local Random Forest Model (Cached)
# -----------------------------
@st.cache_resource
def load_and_train_rf():
    try:
        df = pd.read_csv("fake_reviews_dataset.csv")
        X = df["text"]
        y = df["label"]

        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_tfidf, y)
        
        return vectorizer, rf_model
    except FileNotFoundError:
        return None, None

rf_vectorizer, rf_model = load_and_train_rf()

# -----------------------------
# Groq API Scanner (Consensus Logic)
# -----------------------------
try:
    client = Groq(api_key="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")
except Exception as e:
    st.error(f"Initialization Failed: {str(e)}")
    st.stop()

def analyze_hybrid_consensus(review_text, rf_prediction):
    """
    Blends the local model's output with Groq's language capabilities 
    to make a final consensus prediction.
    """
    rf_status = "REAL/AUTHENTIC" if rf_prediction in ["OR", 1, "real"] else "FAKE/SUSPICIOUS"
    
    system_prompt = (
        "You are a master cybersecurity analysis node synthesizing a hybrid fraud scan.\n"
        f"Our local hardware array flagged this review as mathematically: {rf_status}.\n\n"
        "Analyze the text yourself, cross-reference it with the local hardware flag, and make the FINAL absolute decision. "
        "Your response MUST start with either '[STATUS: REAL]' or '[STATUS: FAKE]'.\n\n"
        "Directly below the status tag, output exactly 3 bullet points:\n"
        "1. Direct confirmation of whether your cloud assessment agreed or disagreed with the local Random Forest.\n"
        "2. A breakdown of the structural anomalies, syntax, or realism gaps found in the text.\n"
        "3. A final risk level score (0% to 100%)."
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review Text to evaluate:\n\"\"\"\n{review_text}\n\"\"\""}
            ],
            temperature=0.1,
            max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"ERROR: Communication breach with Groq API. Details: {str(e)}"

# -----------------------------
# Exact Cyberpunk 2077 Theme CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: #030303;
    background-image:
        linear-gradient(rgba(252, 238, 9, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(252, 238, 9, 0.03) 1px, transparent 1px),
        radial-gradient(circle at 15% 20%, rgba(0, 240, 255, 0.1), transparent 30%),
        radial-gradient(circle at 85% 75%, rgba(255, 0, 85, 0.1), transparent 35%);
    background-size: 20px 20px, 20px 20px, 100% 100%, 100% 100%;
    color: #FFFFFF;
    font-family: 'Courier New', Courier, monospace;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.title {
    font-size: 44px;
    font-weight: 900;
    letter-spacing: 4px;
    color: #FCEE09;
    text-align: center;
    text-shadow: 0 0 10px rgba(252, 238, 9, 0.6), 3px 3px 0px #FF0055;
    margin-bottom: 2px;
}

.subtitle {
    text-align: center;
    font-size: 14px;
    font-weight: bold;
    letter-spacing: 3px;
    color: #00F0FF;
    text-shadow: 0 0 5px rgba(0, 240, 255, 0.5);
    margin-bottom: 40px;
}

.stTextArea label {
    color: #FCEE09 !important;
    font-weight: 800;
    letter-spacing: 1px;
}

.stTextArea div[data-baseweb="textarea"] {
    background: #0a0b0d !important;
    border: 2px solid #00F0FF !important;
    border-radius: 0px !important;
    clip-path: polygon(0 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%);
}

.stTextArea div[data-baseweb="textarea"]:focus-within {
    border: 2px solid #FCEE09 !important;
    box-shadow: 0 0 15px rgba(252, 238, 9, 0.4);
}

textarea {
    background: transparent !important;
    color: #00F0FF !important;
    font-size: 16px;
}

.stButton > button {
    width: 100%;
    background: #FCEE09;
    color: #000000 !important;
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 3px;
    border: none;
    border-radius: 0px;
    padding: 14px;
    clip-path: polygon(10px 0%, 100% 0%, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0% 100%, 0% 10px);
    transition: 0.2s;
    border-right: 4px solid #FF0055;
}

.stButton > button:hover {
    background: #00F0FF !important;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.6);
    border-right: 4px solid #FCEE09;
}

.stSuccess {
    background: rgba(0, 240, 255, 0.07) !important;
    border: 1px solid #00F0FF !important;
    border-left: 6px solid #00F0FF !important;
    color: #00F0FF !important;
    border-radius: 0px;
}

.stError {
    background: rgba(255, 0, 85, 0.07) !important;
    border: 1px solid #FF0055 !important;
    border-left: 6px solid #FF0055 !important;
    color: #FF0055 !important;
    border-radius: 0px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI Components & App Logic
# -----------------------------

st.markdown('<div class="title">🤖 ReviewGuardian AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI BASED REVIEW DETECTION SERVICE</div>', unsafe_allow_html=True)

user_review = st.text_area(
    "PASTE THE REVIEW TEXT BELOW:", 
    placeholder="Type or paste the review you want to analyze...",
    height=150
)

st.write("")
analyze_button = st.button("RUN HYBRID SCAN")

if analyze_button:
    if user_review.strip() == "":
        st.warning("System requires explicit input text before running scanners.")
    else:
        with st.spinner("Synthesizing local array and network vectors..."):
            # Step 1: Query Local Hardware (Random Forest)
            if rf_model is not None and rf_vectorizer is not None:
                vec_input = rf_vectorizer.transform([user_review])
                rf_prediction = rf_model.predict(vec_input)[0]
            else:
                rf_prediction = "unknown"  # Graceful fallback if dataset is missing
            
            # Step 2: Query Cloud Network Matrix with Local Context
            consensus_response = analyze_hybrid_consensus(user_review, rf_prediction)
        
        # Step 3: Output Clean Combined Interface Readout
        if "ERROR" in consensus_response:
            st.error(consensus_response)
        elif "[STATUS: REAL]" in consensus_response:
            clean_text = consensus_response.replace("[STATUS: REAL]", "").strip()
            st.success(f"### ✔ AUTHENTIC REVIEW\n\n{clean_text}")
        else:
            clean_text = consensus_response.replace("[STATUS: FAKE]", "").strip()
            st.error(f"### ❌ SUSPICIOUS TEXT DETECTED\n\n{clean_text}")