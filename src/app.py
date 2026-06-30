import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page Config streamlit
# -----------------------------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Train Model on cache to 
# -----------------------------
@st.cache_resource
def load_and_train_model():
    # Load your dataset
    df = pd.read_csv("fake_reviews_dataset.csv")

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    return vectorizer, model

# Try to load the model, catch error if CSV is missing
try:
    vectorizer, model = load_and_train_model()
except FileNotFoundError:
    st.error("Error: 'fake_reviews_dataset.csv' not found. Please place it in the same directory.")
    st.stop()

# -----------------------------
# Custom Cyberpunk Theme CSS
# -----------------------------
st.markdown("""
<style>
/* BACKGROUND */
.stApp {
    background: #080808;
    background-image:
    radial-gradient(circle at 15% 20%, rgba(0,255,255,.08), transparent 25%),
    radial-gradient(circle at 85% 75%, rgba(255,210,0,.08), transparent 30%);
    color: white;
}

/* Hide Streamlit UI elements safely */
header, footer, #MainMenu {
    visibility: hidden;
}

/* MAIN PANEL CONTAINER */
.cyber-container {
    background: #101010;
    border: 2px solid #FFD400;
    padding: 35px;
    border-radius: 8px;
    box-shadow: 0 0 8px #FFD400, 0 0 30px rgba(255,212,0,.25);
    margin-bottom: 25px;
}

/* TYPOGRAPHY */
.title {
    font-size: 46px;
    font-weight: 900;
    letter-spacing: 3px;
    color: #FFD400;
    text-align: center;
    text-shadow: 0 0 8px #FFD400, 0 0 18px #FFD400;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    letter-spacing: 2px;
    color: #00F5FF;
    margin-bottom: 30px;
}

/* TEXT AREA STYLING */
.stTextArea label {
    color: #FFD400 !important;
    font-weight: bold;
}

.stTextArea div[data-baseweb="textarea"] {
    background: #0d0d0d !important;
    border: 2px solid #00F5FF !important;
    border-radius: 8px !important;
    transition: .3s;
}

.stTextArea div[data-baseweb="textarea"]:focus-within {
    border: 2px solid #FFD400 !important;
    box-shadow: 0 0 15px #FFD400, 0 0 40px rgba(255,212,0,.3);
}

textarea {
    background: transparent !important;
    color: #00F5FF !important;
    font-size: 17px;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* BUTTON STYLING */
.stButton > button {
    width: 100%;
    background: #FFD400;
    color: black !important;
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 2px;
    border: none;
    border-radius: 6px;
    padding: 15px;
    transition: .25s;
    box-shadow: 0 0 10px #FFD400, 0 0 25px rgba(255,212,0,.45);
}

.stButton > button:hover {
    background: #00F5FF !important;
    color: black !important;
    transform: scale(1.02);
    box-shadow: 0 0 10px #00F5FF, 0 0 35px rgba(0,245,255,.6);
}

.stButton > button:active {
    background: #00F5FF !important;
    color: black !important;
}

/* ALERT CUSTOMIZATIONS */
.stSuccess {
    background: #0c1b12 !important;
    border-left: 6px solid #00ff66 !important;
    color: #00ff66 !important;
}

.stError {
    background: #200707 !important;
    border-left: 6px solid #ff003c !important;
    color: #ff4d6d !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #FFD400;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI Components & App Logic
# -----------------------------

# Render Title Header
st.markdown('<div class="title">🤖 FAKE REVIEW DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-POWERED REVIEW DETECTION SERVICE</div>', unsafe_allow_html=True)

# Wrap your input form inside the custom .cyber-container class div
st.markdown('<div class="cyber-container">', unsafe_allow_html=True)

user_review = st.text_area(
    "PASTE THE REVIEW TEXT BELOW:", 
    placeholder="Type or paste the review you want to analyze...",
    height=150
)

# Create space or margin before the action button
st.write("")
analyze_button = st.button("RUN SCANNERS")

st.markdown('</div>', unsafe_allow_html=True)

# Process Action
if analyze_button:
    if user_review.strip() == "":
        st.warning("Please enter some text before running the scan.")
    else:
        # Transform input using the trained TfidfVectorizer
        processed_input = vectorizer.transform([user_review])
        
        # Predict outcome
        prediction = model.predict(processed_input)[0]
        
        # Display custom styled response based on your dataset labels
        # Note: Change 'CG'/'OR' or 1/0 to match the exact labels in your CSV
        if prediction == "OR" or prediction == 1 or str(prediction).lower() == "real":
            st.success("###  Authentic Review\n\nThis text patterns match organic, real-user behaviors.")
        else:
            st.error("### ❌ Suspicious Review\n\nHigh probability of computer-generated or coordinated fraudulent text structure.")

            