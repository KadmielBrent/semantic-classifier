import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import re
import sys


# ----------------------------------------------------------------------
# 0. PATH RESOLUTION (works both in dev and packaged .exe)
# ----------------------------------------------------------------------
def get_base_dir():
    """Return the correct base directory whether running as script or packaged."""
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Running as a normal Python script
        return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()


# ----------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ARCHITECTURE TITLE
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Filipino Fake News Frame Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("Semantic Frame Classifier")
st.caption("BS Data Analytics Thesis Proposed System | University of the Cordilleras")
st.markdown("---")


# ----------------------------------------------------------------------
# 2. CORE TAXONOMY DEFINITIONS
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 3. PATH SETUP (now using BASE_DIR from step 0)
# ----------------------------------------------------------------------
TAGALOG_DIR = BASE_DIR / "final_export_fnf1_tagalog"
ENGLISH_DIR = BASE_DIR / "final_export_fnf2_english"


# ----------------------------------------------------------------------
# 4. HELPER: SEARCH FOR MODEL FILE RECURSIVELY
# ----------------------------------------------------------------------
def find_checkpoint_file_recursive(folder_path: Path):
    if not folder_path.exists() or not folder_path.is_dir():
        return None

    priority_names = [
        "best_model.pt",
        "best_model.pth",
        "pytorch_model.bin",
        "model.pt",
        "model.pth",
        "checkpoint.pt",
        "checkpoint.pth"
    ]

    # First, try preferred filenames recursively
    for name in priority_names:
        matches = list(folder_path.rglob(name))
        if matches:
            return matches[0]

    # Then try any likely model file recursively
    all_possible = []
    for pattern in ["*.pt", "*.pth", "*.bin"]:
        all_possible.extend(folder_path.rglob(pattern))

    if all_possible:
        return all_possible[0]

    return None


# ----------------------------------------------------------------------
# 5. HELPER: SHOW DIRECTORY TREE
# ----------------------------------------------------------------------
def list_all_contents(folder_path: Path):
    try:
        return [str(p.relative_to(folder_path)) for p in folder_path.rglob("*")]
    except Exception:
        return []


# ----------------------------------------------------------------------
# 6. CACHED LOCAL MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource
def load_local_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    try:
        tagalog_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=8,
            problem_type="multi_label_classification"
        )

        english_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=8,
            problem_type="multi_label_classification"
        )

        # Verify folders exist
        if not TAGALOG_DIR.exists():
            st.error("Tagalog model folder missing.")
            st.stop()

        if not ENGLISH_DIR.exists():
            st.error("English model folder missing.")
            st.stop()

        # Find checkpoints
        tagalog_checkpoint = find_checkpoint_file_recursive(TAGALOG_DIR)
        english_checkpoint = find_checkpoint_file_recursive(ENGLISH_DIR)

        if tagalog_checkpoint is None:
            st.error("No valid Tagalog checkpoint found.")
            st.stop()

        if english_checkpoint is None:
            st.error("No valid English checkpoint found.")
            st.stop()

        # Load weights
        tagalog_weights = torch.load(
            str(tagalog_checkpoint),
            map_location="cpu",
            weights_only=True
        )

        english_weights = torch.load(
            str(english_checkpoint),
            map_location="cpu",
            weights_only=True
        )

        # Apply Tagalog weights
        if isinstance(tagalog_weights, dict) and "state_dict" in tagalog_weights:
            tagalog_model.load_state_dict(
                tagalog_weights["state_dict"],
                strict=False
            )
        elif isinstance(tagalog_weights, dict):
            tagalog_model.load_state_dict(
                tagalog_weights,
                strict=False
            )
        else:
            st.error("Unexpected Tagalog checkpoint format.")
            st.stop()

        # Apply English weights
        if isinstance(english_weights, dict) and "state_dict" in english_weights:
            english_model.load_state_dict(
                english_weights["state_dict"],
                strict=False
            )
        elif isinstance(english_weights, dict):
            english_model.load_state_dict(
                english_weights,
                strict=False
            )
        else:
            st.error("Unexpected English checkpoint format.")
            st.stop()

    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

    tagalog_model.eval()
    english_model.eval()

    return tokenizer, tagalog_model, english_model


# ----------------------------------------------------------------------
# 7. LOAD MODELS
# ----------------------------------------------------------------------
tokenizer, tagalog_model, english_model = load_local_pipeline()


# ----------------------------------------------------------------------
# 8. SIDEBAR MANAGEMENT & STUDY BENCHMARKS
# ----------------------------------------------------------------------
st.sidebar.header("🔧 Model Configuration")

threshold = st.sidebar.slider(
    "Classification Gating Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
    help="The sigmoid probability boundary where a semantic frame is predicted as present."
)

st.sidebar.markdown("### 🧠 Semantic Frame Definitions")

for frame, description in FRAME_DESC.items():
    st.sidebar.markdown(
        f"**{frame.replace('_', ' ').title()}**\n"
        f"<div style='font-size:13px; margin-bottom:10px;'>"
        f"{description}"
        f"</div>",
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------
# 9. LANGUAGE SELECTION TOGGLE (REPLACES AUTO-DETECTION)
# ----------------------------------------------------------------------
def language_selector():
    """
    Display a toggle button for language selection.
    Returns 'tagalog' or 'english'.
    """
    st.subheader("🌐 Select Model")
    
    # Create two columns for the toggle buttons
    col1, col2 = st.columns(2)
    
    # Initialize session state for language if not exists
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'tagalog'
    
    # Tagalog button
    with col1:
        if st.button(
            "🇵🇭 Tagalog", 
            key="tagalog_btn",
            use_container_width=True,
            type="primary" if st.session_state.selected_language == 'tagalog' else "secondary"
        ):
            st.session_state.selected_language = 'tagalog'
            st.rerun()
    
    # English button
    with col2:
        if st.button(
            "🇺🇸 English", 
            key="english_btn",
            use_container_width=True,
            type="primary" if st.session_state.selected_language == 'english' else "secondary"
        ):
            st.session_state.selected_language = 'english'
            st.rerun()
    
    # Display current selection with styling
    current_lang = st.session_state.selected_language
    lang_display = "🇵🇭 Tagalog" if current_lang == 'tagalog' else "🇺🇸 English"
    model_badge = "🔴 FNF1" if current_lang == 'tagalog' else "🔵 FNF2"
    
    
    return st.session_state.selected_language


# ----------------------------------------------------------------------
# 10. MULTI-LABEL INFERENCE ENGINE
# ----------------------------------------------------------------------
def run_inference(text, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().tolist()

    if isinstance(probabilities, float):
        probabilities = [probabilities]

    return probabilities


# ----------------------------------------------------------------------
# 11. INPUT INTERFACE
# ----------------------------------------------------------------------

# Display language selector above the text area
selected_language = language_selector()

st.subheader("📝 Input News Content for Narrative Evaluation")
user_text = st.text_area(
    "Paste the news article body or sentence segment below:",
    height=150,
    placeholder="E.g., Ang mga corrupt na politiko ay patuloy na nagtatago ng yaman..."
)


# ----------------------------------------------------------------------
# 12. MODEL OUTPUT - SHOW ONLY ACTIVE FRAMES IN DESCENDING ORDER

# ----------------------------------------------------------------------
if st.button("Analyze", type="primary"):
    if user_text.strip():
        # Use the selected language from session state
        detected_language = selected_language
        
        # Create header with language and model info
        if detected_language == "tagalog":
            model_badge = "🔴 FNF1 - Tagalog Model"
            header_color = "#FF6B6B"
        else:
            model_badge = "🔵 FNF2 - English Model"
            header_color = "#4D9DE0"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {header_color}20 0%, {header_color}05 100%); 
                    padding: 1rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;
                    border-left: 4px solid {header_color};">
            <h3 style="margin: 0; color: {header_color};">📊 Classification Results</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">🌐 Language: {detected_language.upper()} | {model_badge}</p>
        </div>
        """, unsafe_allow_html=True)
        
        selected_model = tagalog_model if detected_language == "tagalog" else english_model
        
        with st.spinner("🔍 Analyzing semantic frames..."):
            probs = run_inference(user_text, selected_model)
            
            if len(probs) != len(LABELS):
                st.error(f"Output size mismatch. Expected {len(LABELS)} labels, got {len(probs)}.")
            else:
                # Create list of (label, probability) pairs
                frame_predictions = list(zip(LABELS, probs))
                
                # Filter only active frames (probability >= threshold)
                active_frames = [(label, float(prob)) for label, prob in frame_predictions if float(prob) >= threshold]
                
                # Sort active frames by probability in descending order
                active_frames.sort(key=lambda x: x[1], reverse=True)
                
                if not active_frames:
                    # No active frames found
                    st.warning("⚠️ No semantic frames exceeded the classification threshold. Try lowering the threshold in the sidebar.")
                    
                    # Show the highest probability frame as a suggestion
                    highest_frame = max(frame_predictions, key=lambda x: x[1])
                    st.info(f"💡 Highest scoring frame: **{highest_frame[0].replace('_', ' ').title()}** at {highest_frame[1]:.1%} (below threshold of {threshold:.0%})")
                else:
                    # Display active frames in descending order with a counter
                    st.markdown(f"""
                    <div style="background: #28a74515; 
                                padding: 0.5rem 1rem; 
                                border-radius: 8px; 
                                margin-bottom: 1rem;
                                border-left: 4px solid #28a745;">
                        <span style="font-weight: 600; color: #28a745;">✅ {len(active_frames)} Active Frame{'s' if len(active_frames) > 1 else ''} Detected</span>
                        <span style="margin-left: 1rem; font-size: 0.9rem; opacity: 0.7;">
                            (Sorted by confidence: highest to lowest)
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display active frames in descending order
                    for rank, (label, prob) in enumerate(active_frames, 1):
                        # Determine color based on frame category
                        frame_colors = {
                            "political_conflict": "#FF6B6B",
                            "scandal_corruption": "#FF8C42",
                            "conspiracy_hidden_agenda": "#F9D56E",
                            "victimization": "#4D9DE0",
                            "heroism": "#48A9A6",
                            "threat_danger": "#E15554",
                            "economic_impact": "#3BB273",
                            "nationalism_patriotism": "#9C89B8"
                        }
                        color = frame_colors.get(label, "#666666")
                        
                        # Create card-style display with rank indicator
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color}20 0%, {color}05 100%); 
                                    padding: 1rem; 
                                    border-radius: 10px; 
                                    margin-bottom: 0.8rem;
                                    border-left: 4px solid {color};
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                    transition: all 0.3s ease;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="display: flex; align-items: center;">
                                    <span style="display: inline-flex; 
                                               align-items: center; 
                                               justify-content: center;
                                               background: {color};
                                               color: white;
                                               border-radius: 50%;
                                               width: 28px;
                                               height: 28px;
                                               font-size: 0.8rem;
                                               font-weight: 700;
                                               margin-right: 12px;">
                                        {rank}
                                    </span>
                                    <span style="font-weight: 600; font-size: 1.05rem; color: {color};">
                                        {label.replace('_', ' ').title()}
                                    </span>
                                </div>
                                <span style="font-size: 1.1rem; color: {color}; font-weight: 700;">
                                    {prob:.1%}
                                </span>
                            </div>
                            <div style="margin-top: 0.7rem;">
                                <div style="background: #e9ecef; border-radius: 6px; height: 8px; overflow: hidden;">
                                    <div style="width: {prob*100}%; 
                                                height: 100%; 
                                                background: {color}; 
                                                border-radius: 6px; 
                                                transition: width 0.5s ease;">
                                    </div>
                                </div>
                            </div>
                            <div style="margin-top: 0.5rem; display: flex; justify-content: space-between;">
                                <span style="font-size: 0.75rem; color: #6c757d;">
                                    {FRAME_DESC.get(label, '')[:100]}...
                                </span>
                                <span style="font-size: 0.75rem; color: {color}; font-weight: 500;">
                                    Confidence: {prob:.1%}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please input text before execution.")