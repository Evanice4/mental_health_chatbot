import gradio as gr
import pandas as pd
import numpy as np
import re, nltk, warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException
warnings.filterwarnings("ignore")

for r in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(r, quiet=True)

# ── Load dataset ──
df = pd.read_csv("combined_english_arabic_dataset.csv")

def detect_column(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

qcol = detect_column(df, ["question","input","text","prompt","Question","Input","Text"])
acol = detect_column(df, ["answer","response","output","label","Answer","Response","Output"])
df = df.rename(columns={qcol: "question", acol: "answer"})
df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)

# ── Preprocessing ──
lemmatizer = WordNetLemmatizer()
stop_en = set(stopwords.words("english"))

def clean_text(text, lang="en"):
    if not isinstance(text, str):
        return ""
    if lang not in ["ar", "arabic"]:
        text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if lang not in ["ar", "arabic"]:
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_en and len(t) > 2]
    return " ".join(tokens)

def safe_detect(t):
    try:
        return detect(str(t))
    except:
        return "en"

df["cleaned"] = df["question"].apply(clean_text)

# ── TF-IDF model ──
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
mat = tfidf.fit_transform(df["cleaned"].fillna(""))

# ── Crisis detection ──
CRISIS_KW = ["suicide", "kill myself", "end my life", "self harm", "want to die", "cut myself"]
CRISIS_MSG = (
    " **Your safety matters.**\n\n"
    "If you are in crisis, please reach out:\n"
    "-  https://www.befrienders.org\n"
    "-  Crisis Text Line: text HOME to 741741\n"
    "-  https://findahelpline.com\n\n"
    "You are not alone. "
)

# ── Response function ──
def respond(user_msg, history, show_score):
    if not user_msg.strip():
        return history, ""
    
    if any(k in user_msg.lower() for k in CRISIS_KW):
        history.append({"role": "user", "content": user_msg})
        history.append({"role":"assistant", "content": CRISIS_MSG})
        return history, ""
    
    lang = safe_detect(user_msg)
    clean = clean_text(user_msg, lang)
    
    if not clean:
        reply = "I'm here to listen. Could you share more about how you're feeling?"
    else:
        vec = tfidf.transform([clean])
        sims = cosine_similarity(vec, mat).flatten()
        idx = sims.argmax()
        score = sims[idx]
        reply = df["answer"].iloc[idx] if score > 0.05 else "I understand. Could you tell me more about how you are feeling?"
        if show_score:
            reply += f"\n\n_Confidence: {score:.3f}_"
    
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": str(reply)})
    return history, ""

STARTERS = [
    "I've been feeling really anxious lately",
    "I can't seem to stop worrying about everything",
    "I feel overwhelmed with work and life",
    "I've been having trouble sleeping",
    "I feel lonely and disconnected",
    "How can I manage my stress better?",
]

# ── CSS ──
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
.gradio-container, body {
    font-family: 'Nunito', sans-serif !important;
    background: #f0f2ff !important;
    color: #1a1a3a !important;
}
.app-header {
    text-align: center;
    padding: 22px 20px 14px;
    background: linear-gradient(135deg, #5b6ef5 0%, #9b59f5 100%);
    border-radius: 14px;
    margin-bottom: 16px;
    box-shadow: 0 6px 22px rgba(91,110,245,0.28);
}
.app-header h1 { font-size: 1.9rem; font-weight: 700; color: #ffffff !important; margin: 0 0 5px; }
.app-header p  { color: rgba(255,255,255,0.92) !important; font-size: 0.93rem; margin: 0; }
#chatbot {
    background: #ffffff !important;
    border: 1.5px solid #cdd0ff !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 14px rgba(91,110,245,0.10) !important;
}
.sidebar-panel {
    background: #ffffff !important;
    border: 1.5px solid #cdd0ff !important;
    border-radius: 12px !important;
    padding: 14px 15px !important;
    margin-bottom: 12px !important;
}
.sidebar-panel p, .sidebar-panel li, .sidebar-panel h3,
.sidebar-panel strong, .sidebar-panel span {
    color: #1a1a3a !important;
}
.gradio-container .markdown-body,
.gradio-container .markdown-body p,
.gradio-container .markdown-body li,
.gradio-container .markdown-body strong {
    color: #1a1a3a !important;
}
#msg-input textarea {
    background: #ffffff !important;
    color: #1a1a3a !important;
    border: 2px solid #c5c9f8 !important;
    border-radius: 11px !important;
    font-size: 0.95rem !important;
}
#msg-input textarea::placeholder { color: #8a90cc !important; }
button.primary { background: linear-gradient(135deg,#5b6ef5,#9b59f5) !important; color:#fff !important; border:none !important; border-radius:10px !important; font-weight:600 !important; }
button.secondary { background:#fff !important; color:#5b6ef5 !important; border:2px solid #c5c9f8 !important; border-radius:10px !important; font-weight:600 !important; }
.disclaimer {
    background: #fff7ed !important;
    border-left: 4px solid #f59b2b !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 11px 14px !important;
    font-size: 0.82rem !important;
    margin-top: 10px !important;
}
.disclaimer, .disclaimer p, .disclaimer strong { color: #6b4a18 !important; }
"""

WELCOME_MSG = (
    "###  Welcome to Mindful\n"
    "I'm here to listen and support you.\n\n"
    "**Try asking about:**\n"
    "- Anxiety or stress\n"
    "- Depression or low mood\n"
    "- Sleep difficulties\n"
    "- Relationship issues\n"
    "- General emotional support"
)

DISCLAIMER_TEXT = (
    " **Disclaimer:** "
    "This is **not** a substitute for professional mental health care. "
    "In a crisis, please contact a licensed professional or emergency services."
)

# ── Build UI ──
with gr.Blocks(css=CSS, title=" MindFul Chatbot", theme=gr.themes.Default()) as demo:

    gr.HTML("""
    <div class="app-header">
        <h1> MindFul — Mental Health Support Chatbot</h1>
        <p>A compassionate AI companion &nbsp;·&nbsp; Bilingual: English &amp; Arabic</p>
    </div>""")

    with gr.Row():
        # ── Sidebar ──
        with gr.Column(scale=1, min_width=230):
            gr.Markdown(WELCOME_MSG, elem_classes=["sidebar-panel"])

            gr.Markdown("###  Settings")
            show_score = gr.Checkbox(
                label="Show confidence score",
                value=False,
                info="Display how confident the model is"
            )
            starter_btn = gr.Button(" Suggest a topic", variant="secondary")

            gr.Markdown(
                f"###  Dataset Info\n"
                f"- **Entries:** {len(df):,}\n"
                f"- **Languages:** 2,\n"
                f"- **Vocabulary:** {len(tfidf.vocabulary_):,} terms\n"
                f"- **Model:** TF-IDF + Cosine Similarity, (LLM) DistilGPT2",
                elem_classes=["sidebar-panel"]
            )

            gr.Markdown(DISCLAIMER_TEXT, elem_classes=["disclaimer"])

        # ── Chat area ──
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                elem_id="chatbot",
                height=480
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Share how you're feeling... (English or Arabic)",
                    lines=2,
                    max_lines=4,
                    scale=5,
                    elem_id="msg-input",
                    show_label=False
                )
                with gr.Column(scale=1, min_width=110):
                    send_btn  = gr.Button("Send ",  variant="primary",   size="lg")
                    clear_btn = gr.Button("Clear ", variant="secondary", size="sm")

            gr.Examples(
                examples=[
                    "I feel very anxious and can't stop worrying",
                    "I've been depressed and don't know what to do",
                    "I can't sleep at night, my mind won't stop",
                    "I feel overwhelmed and stressed with everything",
                    "أشعر بالقلق الشديد ولا أعرف ماذا أفعل",
                ],
                inputs=msg,
                label=" Try these example messages:"
            )

    # ── Events ──
    send_btn.click(fn=respond, inputs=[msg, chatbot_ui, show_score], outputs=[chatbot_ui, msg])
    msg.submit(fn=respond, inputs=[msg, chatbot_ui, show_score], outputs=[chatbot_ui, msg])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot_ui, msg])
    starter_btn.click(fn=lambda: str(np.random.choice(STARTERS)), outputs=msg)

demo.launch()