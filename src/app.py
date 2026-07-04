import streamlit as st
import os
import pandas as pd
import numpy as np
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
# Train Multi-Algorithm Array (Cached)
# -----------------------------
@st.cache_resource
def train_ensemble_system():
    """
    Trains both Logistic Regression and Random Forest models on the dataset
    to provide multi-layered algorithmic baseline classification probabilities.
    """
    try:
        df = pd.read_csv("fake_reviews_dataset.csv")
        X = df["text"]
        y = df["label"]

        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)

        # Algorithm 1: Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_tfidf, y)

        # Algorithm 2: Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_tfidf, y)
        
        return vectorizer, lr_model, rf_model
    except FileNotFoundError:
        return None, None, None

vectorizer, lr_model, rf_model = train_ensemble_system()

# -----------------------------
# Initialize Groq Client
# -----------------------------
try:
    client = Groq(api_key="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")
except Exception as e:
    st.error(f"Initialization Failed: {str(e)}")
    st.stop()

def analyze_hybrid_system(review_text, lr_prob_fake, rf_prob_fake):
    """
    Fuses textual analysis with numeric vectors from local traditional algorithms 
    to yield a comprehensive cybernetic assessment breakdown.
    """
    system_prompt = (
        "You are an advanced AI fraud detection node checking user text validation signatures.\n"
        f"Our local hardware array reports these mathematical indicators:\n"
        f"- Logistic Regression Node Prob(Fake): {lr_prob_fake:.2%}\n"
        f"- Random Forest Node Prob(Fake): {rf_prob_fake:.2%}\n\n"
        "Analyze the text yourself. Combine your linguistic findings with the traditional model stats to output your breakdown.\n\n"
        "Your response MUST adhere strictly to this custom layout template:\n"
        "[CONFIDENCE: XX.XX%]\n"
        "[STATUS: REAL or FAKE]\n"
        "1. Classification Reasoning: (Provide deep reasoning)\n"
        "2. Suspicious Writing Patterns: (List bullet points on syntax quirks or artificial anomalies)\n"
        "3. Final Verdict: (Summarize consensus resolution details)"
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Target Review Data:\n\"\"\"\n{review_text}\n\"\"\""}
            ],
            temperature=0.1,
            max_tokens=450
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"ERROR: Communication breach with Groq API. Details: {str(e)}"

# -----------------------------
# Exact Cyberpunk 2077 Theme CSS
# -----------------------------
st.markdown("""
<style>
/* BACKGROUND & GLOBAL GLITCH STYLE */
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

/* TYPOGRAPHY */
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

/* INPUT SECTION STYLE */
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
    transition: 0.3s ease-in-out;
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

/* ACTION BUTTON */
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

/* DISPLAY INTERFACE PLUGINS */
.confidence-title {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 15px;
}

.confidence-value {
    color: #FCEE09;
    font-size: 38px;
    font-weight: 900;
    text-shadow: 0 0 8px rgba(252, 238, 9, 0.5);
    margin-bottom: 20px;
}

/* STREAMLIT NATIVE EXPANDER OVERRIDES FOR CYBERPUNK THEME */
.stDetails {
    background: #0a0b0d !important;
    border: 1px solid #FF0055 !important;
    border-radius: 0px !important;
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
st.markdown('<div class="subtitle">AI BASED REVIEW DETECTION SYSTEM</div>', unsafe_allow_html=True)

user_review = st.text_area(
    "Paste review text below:", 
    placeholder="Type or paste the review you want to analyze...",
    height=150
)

st.write("")
analyze_button = st.button("RUN SCANNER")

if analyze_button:
    if user_review.strip() == "":
        st.warning("System requires explicit input text before running scanners.")
    else:
        with st.spinner("Executing multi-model matrix alignment checks..."):
            # Step 1: Default baseline probability states if local dataset files are unavailable
            lr_prob_fake, rf_prob_fake = 0.50, 0.50
            
            if vectorizer is not None and lr_model is not None and rf_model is not None:
                token_vector = vectorizer.transform([user_review])
                
                # Extract probabilities for the 'Fake'/'CG' signature class matching standard dataset targets
                classes = list(lr_model.classes_)
                fake_index = classes.index("CG") if "CG" in classes else (1 if 1 in classes else 0)
                
                lr_prob_fake = lr_model.predict_proba(token_vector)[0][fake_index]
                rf_prob_fake = rf_model.predict_proba(token_vector)[0][fake_index]
            
            # Step 2: Route local probability maps through Groq's high-level semantic refinery
            raw_response = analyze_hybrid_system(user_review, lr_prob_fake, rf_prob_fake)
        
        if "ERROR" in raw_response:
            st.error(raw_response)
        else:
            # Step 3: Parsing tags from the custom output layout template safely
            confidence_val = "00.00%"
            verdict_status = "FAKE"
            clean_display_text = raw_response

            try:
                if "[CONFIDENCE:" in raw_response:
                    confidence_val = raw_response.split("[CONFIDENCE:")[1].split("]")[0].strip()
                    clean_display_text = clean_display_text.replace(f"[CONFIDENCE: {confidence_val}]", "").replace(f"[CONFIDENCE:{confidence_val}]", "")
                
                if "[STATUS:" in raw_response:
                    verdict_status = raw_response.split("[STATUS:")[1].split("]")[0].strip()
                    clean_display_text = clean_display_text.replace(f"[STATUS: {verdict_status}]", "").replace(f"[STATUS:{verdict_status}]", "")
            except Exception:
                pass # Fallback parsing if formatting contains standard string anomalies

            # Step 4: UI Rendering matching WhatsApp Image 2026-07-04 at 3.39.37 PM.jpeg layout
            if "REAL" in verdict_status:
                st.success("### ✔ Authentic Review\n\nText patterns confirm organic, verified consumer activity.")
            else:
                st.error("### ❌ Suspicious Review\n\nHigh probability of computer-generated or coordinated fraudulent text structure.")
            
            # Prominent Confidence Metrics
            st.markdown('<div class="confidence-title">System Analysis Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-value">{confidence_val}</div>', unsafe_allow_html=True)
            
            # Interactive Information Dropdown (Expander Module)
            with st.expander("👁 AI Explanation", expanded=True):
                st.markdown(clean_display_text.strip())