import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page config
st.set_page_config(
    page_title="Filipino Fake News Frame Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Semantic Frame Classifier")
st.caption("BS Data Analytics Thesis Proposed System | University of the Cordilleras")
st.markdown("---")

# Your Hugging Face repositories
HUGGINGFACE_REPO = {
    "tagalog": "KadmielBrent/tagalog_model",
    "english": "KadmielBrent/english_model"  # Make sure this one is fixed too!
}

# Labels and descriptions
LABELS = [
    "political_conflict",
    "scandal_corruption",
    "conspiracy_hidden_agenda",
    "victimization",
    "heroism",
    "threat_danger",
    "economic_impact",
    "nationalism_patriotism"
]

FRAME_DESC = {
    "political_conflict": "Frames events as a binary struggle or antagonism between opposing political groups or factions.",
    "scandal_corruption": "Highlights wrongdoing, unethical behavior, or violations of values to delegitimize individuals or institutions.",
    "conspiracy_hidden_agenda": "Alleges the existence of secret plots by influential figures or foreign agents to manipulate public events.",
    "victimization": "Recasts powerful figures as misunderstood victims of an elite establishment to foster public relatability.",
    "heroism": "Depicts protagonists as saviors, truth crusaders, or champions of the people to manufacture legitimacy.",
    "threat_danger": "Narratives designed to trigger fear or anxiety by emphasizing risks to social order, security, or stability.",
    "economic_impact": "Focuses on financial consequences, salvation, or national wealth, often utilizing myths to address economic anxieties.",
    "nationalism_patriotism": "Mobilizes support by appealing to national identity or discrediting critics as 'anti-Filipino'."
}

# Cache model loading
@st.cache_resource
def load_model(language="tagalog"):
    """Load model and tokenizer from Hugging Face."""
    repo = HUGGINGFACE_REPO[language]
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForSequenceClassification.from_pretrained(
            repo,
            num_labels=8,
            problem_type="multi_label_classification"
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading {language} model: {e}")
        st.stop()

# Load both models
with st.spinner("Loading models from Hugging Face..."):
    tagalog_tokenizer, tagalog_model = load_model("tagalog")
    english_tokenizer, english_model = load_model("english")

# Sidebar
st.sidebar.header("🔧 Model Configuration")
threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("### 🧠 Semantic Frame Definitions")
for frame, description in FRAME_DESC.items():
    st.sidebar.markdown(f"**{frame.replace('_', ' ').title()}**\n{description}\n", unsafe_allow_html=True)

# Language selector
def language_selector():
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'tagalog'
    
    st.subheader("🌐 Select Model")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇵🇭 Tagalog", use_container_width=True, 
                     type="primary" if st.session_state.selected_language == 'tagalog' else "secondary"):
            st.session_state.selected_language = 'tagalog'
            st.rerun()
    with col2:
        if st.button("🇺🇸 English", use_container_width=True,
                     type="primary" if st.session_state.selected_language == 'english' else "secondary"):
            st.session_state.selected_language = 'english'
            st.rerun()
    return st.session_state.selected_language

selected_language = language_selector()

# Input
st.subheader("📝 Input News Content")
user_text = st.text_area(
    "Paste the news article below:",
    height=150,
    placeholder="E.g., Ang mga corrupt na politiko ay patuloy na nagtatago ng yaman..."
)

# Inference
def run_inference(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    return probs if isinstance(probs, list) else [probs]

# Analyze
if st.button("Analyze", type="primary"):
    if not user_text.strip():
        st.warning("⚠️ Please input text before execution.")
    else:
        selected_model = tagalog_model if selected_language == "tagalog" else english_model
        selected_tokenizer = tagalog_tokenizer if selected_language == "tagalog" else english_tokenizer
        model_badge = "🔴 FNF1" if selected_language == "tagalog" else "🔵 FNF2"
        header_color = "#FF6B6B" if selected_language == "tagalog" else "#4D9DE0"
        
        st.markdown(f"""
        <div style="background: {header_color}20; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid {header_color};">
            <h3 style="margin: 0; color: {header_color};">📊 Classification Results</h3>
            <p style="margin: 0.5rem 0 0 0;">🌐 Language: {selected_language.upper()} | {model_badge}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Analyzing..."):
            probs = run_inference(user_text, selected_model, selected_tokenizer)
            predictions = list(zip(LABELS, probs))
            active_frames = [(label, float(prob)) for label, prob in predictions if prob >= threshold]
            active_frames.sort(key=lambda x: x[1], reverse=True)
            
            if not active_frames:
                highest = max(predictions, key=lambda x: x[1])
                st.warning(f"No frames exceeded threshold. Highest: {highest[0].replace('_', ' ').title()} ({highest[1]:.1%})")
            else:
                st.success(f"✅ {len(active_frames)} Active Frame{'s' if len(active_frames) > 1 else ''} Detected")
                for rank, (label, prob) in enumerate(active_frames, 1):
                    color = {
                        "political_conflict": "#FF6B6B",
                        "scandal_corruption": "#FF8C42",
                        "conspiracy_hidden_agenda": "#F9D56E",
                        "victimization": "#4D9DE0",
                        "heroism": "#48A9A6",
                        "threat_danger": "#E15554",
                        "economic_impact": "#3BB273",
                        "nationalism_patriotism": "#9C89B8"
                    }.get(label, "#666666")
                    
                    st.markdown(f"""
                    <div style="background: {color}20; padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem; border-left: 4px solid {color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="background: {color}; color: white; border-radius: 50%; padding: 2px 8px; margin-right: 8px;">{rank}</span>
                                <span style="font-weight: 600; color: {color};">{label.replace('_', ' ').title()}</span>
                            </div>
                            <span style="font-weight: 700; color: {color};">{prob:.1%}</span>
                        </div>
                        <div style="background: #e9ecef; border-radius: 6px; height: 8px; margin-top: 8px;">
                            <div style="width: {prob*100}%; height: 100%; background: {color}; border-radius: 6px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
