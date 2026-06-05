import streamlit as st
import whisper
import tempfile
import os
from transformers import BartForConditionalGeneration, BartTokenizer
import re

# Page config
st.set_page_config(
    page_title="AI Meeting Summarizer",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    .main-header h1 { color: white; font-size: 3em; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.2em; margin-top: 10px; }
    .card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
    }
    .section-header {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .summary-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 20px;
        color: white;
        font-size: 1.05em;
        line-height: 1.7;
    }
    .action-point {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid #f5576c;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 8px 0;
        color: white;
    }
    .step-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 15px 20px;
        margin: 8px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(102,126,234,0.4);
    }
    .stat-number { font-size: 2.5em; font-weight: bold; color: #667eea; }
    .stat-label { color: rgba(255,255,255,0.7); font-size: 0.9em; }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 35px !important;
        font-size: 1.1em !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
        width: 100% !important;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(102,126,234,0.5) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] { color: white !important; border-radius: 8px !important; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #f093fb, #f5576c) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 35px !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎙️ AI Meeting Summarizer</h1>
    <p>Transform any meeting audio or video into smart summaries & action points using AI</p>
    <p style="font-size:0.9em; opacity:0.7;">Powered by OpenAI Whisper • Facebook BART • Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-number">🎧</div><div class="stat-label">Audio Transcription</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-number">🤖</div><div class="stat-label">AI Summarization</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-number">✅</div><div class="stat-label">Action Extraction</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><div class="stat-number">📥</div><div class="stat-label">Download Results</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_bart_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

def summarize_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=200, min_length=50,
        length_penalty=2.0, num_beams=4, early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_youtube_id(url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def process_audio(audio_path):
    # Check file size
    size = os.path.getsize(audio_path)
    if size < 1000:
        raise ValueError("Audio file is too small or empty. Please try a different video.")
    model = load_whisper_model()
    result = model.transcribe(audio_path)
    return result["text"]

def analyze_transcript(transcript):
    tokenizer, bart_model = load_bart_model()
    words = transcript.split()
    chunks = [" ".join(words[i:i+800]) for i in range(0, len(words), 800)]
    summaries = []
    for chunk in chunks:
        if len(chunk.strip()) > 50:
            summary = summarize_text(chunk, tokenizer, bart_model)
            summaries.append(summary)
    full_summary = " ".join(summaries)

    action_keywords = [
        "will", "should", "must", "need to", "action", "follow up",
        "complete", "submit", "send", "review", "schedule", "plan",
        "prepare", "discuss", "update", "assign", "deadline", "task",
        "let's", "let us", "going to", "i'll", "we'll", "can help"
    ]
    sentences = transcript.replace("?", ".").replace("!", ".").split(".")
    action_points = []
    for sentence in sentences:
        sentence = sentence.strip()
        if any(keyword in sentence.lower() for keyword in action_keywords):
            if len(sentence) > 20:
                action_points.append(sentence)
    return full_summary, action_points[:8]

def show_results(transcript, full_summary, action_points):
    st.markdown('<div class="section-header">📝 Transcript</div>', unsafe_allow_html=True)
    with st.expander("View Full Transcript", expanded=False):
        st.markdown(f'<div class="card" style="color:rgba(255,255,255,0.85); line-height:1.8;">{transcript}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Meeting Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{full_summary}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">✅ Action Points</div>', unsafe_allow_html=True)
    if action_points:
        for i, point in enumerate(action_points, 1):
            st.markdown(f'<div class="action-point">🔹 <b>Action {i}:</b> {point}</div>', unsafe_allow_html=True)
    else:
        st.warning("No specific action points detected.")

    st.markdown("<br>", unsafe_allow_html=True)
    words = len(transcript.split())
    sentences = len(transcript.split('.'))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{words}</div><div class="stat-label">Words Transcribed</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{sentences}</div><div class="stat-label">Sentences</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(action_points)}</div><div class="stat-label">Action Points</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("✅ Analysis Complete!")

    output = f"""AI MEETING SUMMARIZER - RESULTS
================================
SUMMARY:
{full_summary}

ACTION POINTS:
{chr(10).join([f'{i+1}. {p}' for i, p in enumerate(action_points)])}

FULL TRANSCRIPT:
{transcript}
"""
    st.download_button(
        label="📥 Download Full Results",
        data=output,
        file_name="meeting_summary.txt",
        mime="text/plain"
    )

# Tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🔗 YouTube / URL Link", "ℹ️ How It Works"])

# Tab 1: File Upload
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📂 Upload Your Meeting Audio or Video")
    st.markdown("Supports **MP3, MP4, WAV, M4A, OGG** • Max 200MB • Best for files under 30 minutes")
    uploaded_file = st.file_uploader("Choose a file", type=["mp3", "mp4", "wav", "m4a", "ogg"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.audio(uploaded_file)
        size_mb = round(uploaded_file.size/1024/1024, 2)
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({size_mb} MB)")

        # Quick mode option
        quick_mode = st.checkbox("⚡ Quick Mode (first 5 minutes only — faster results)", value=False)

        if st.button("🚀 Analyze Meeting", key="analyze_file"):
            progress = st.progress(0, text="Starting analysis...")

            with st.spinner("🎧 Step 1/3 — Transcribing audio..."):
                progress.progress(20, text="🎧 Transcribing audio...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    transcript = process_audio(tmp_path)
                    os.unlink(tmp_path)
                    if quick_mode:
                        words = transcript.split()
                        transcript = " ".join(words[:500])
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")
                    os.unlink(tmp_path)
                    st.stop()

            progress.progress(60, text="🤖 Generating summary...")
            with st.spinner("🤖 Step 2/3 — Generating summary..."):
                full_summary, action_points = analyze_transcript(transcript)

            progress.progress(90, text="✅ Extracting action points...")
            progress.progress(100, text="✅ Done!")

            show_results(transcript, full_summary, action_points)

# Tab 2: YouTube
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔗 Paste a YouTube or Video URL")
    st.markdown("💡 **Tip:** For best results, use videos **under 10 minutes**. Longer videos take more time to process.")
    url_input = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if url_input:
        yt_id = extract_youtube_id(url_input)
        if yt_id:
            st.markdown(f"""
            <div class="card">
                <iframe width="100%" height="350"
                src="https://www.youtube.com/embed/{yt_id}"
                frameborder="0" allowfullscreen style="border-radius:10px;">
                </iframe>
            </div>
            """, unsafe_allow_html=True)
            st.success("✅ YouTube video detected!")

            quick_yt = st.checkbox("⚡ Quick Mode (first 5 minutes only — much faster!)", value=True, key="quick_yt")

            if st.button("🚀 Analyze This Video", key="analyze_yt"):
                progress = st.progress(0, text="Starting...")
                try:
                    import yt_dlp
                    with st.spinner("⬇️ Step 1/3 — Downloading audio from YouTube..."):
                        progress.progress(15, text="⬇️ Downloading audio from YouTube...")
                        with tempfile.TemporaryDirectory() as tmpdir:
                            ydl_opts = {
                                'format': 'bestaudio/best',
                                'outtmpl': os.path.join(tmpdir, "audio.%(ext)s"),
                                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
                                'quiet': True,
                            }
                            if quick_yt:
                                ydl_opts['postprocessor_args'] = ['-t', '300']  # 5 min limit

                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                ydl.download([url_input])

                            mp3_files = [f for f in os.listdir(tmpdir) if f.endswith('.mp3')]
                            if not mp3_files:
                                st.error("❌ Could not download audio. This video may be restricted or unavailable. Please try uploading the file directly.")
                                st.stop()

                            audio_path = os.path.join(tmpdir, mp3_files[0])
                            file_size = os.path.getsize(audio_path)

                            if file_size < 1000:
                                st.error("❌ Downloaded audio is empty. This video may be restricted. Please try another video or upload the file directly.")
                                st.stop()

                            progress.progress(40, text="🎧 Transcribing audio...")
                            with st.spinner("🎧 Step 2/3 — Transcribing..."):
                                transcript = process_audio(audio_path)

                            progress.progress(70, text="🤖 Generating summary...")
                            with st.spinner("🤖 Step 3/3 — Generating summary..."):
                                full_summary, action_points = analyze_transcript(transcript)

                            progress.progress(100, text="✅ Done!")
                            show_results(transcript, full_summary, action_points)

                except ImportError:
                    st.error("❌ yt-dlp not found. Please run: pip install yt-dlp")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}\n\nPlease try a different video or upload the file directly.")
        else:
            st.warning("⚠️ Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=...)")

# Tab 3: How It Works
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 How AI Meeting Summarizer Works")
    steps = [
        ("🎙️", "Upload Audio/Video", "Upload your meeting recording in any common format or paste a YouTube link"),
        ("🎧", "Speech to Text (Whisper AI)", "OpenAI Whisper transcribes the audio to text with high accuracy"),
        ("🤖", "AI Summarization (BART)", "Facebook's BART model generates a concise, accurate summary"),
        ("✅", "Action Point Extraction", "Key action items and tasks are automatically identified"),
        ("📥", "Download Results", "Download the complete summary, action points, and transcript"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <span style="font-size:2em">{icon}</span>
            <div style="margin-left:15px;">
                <b style="color:#667eea">{title}</b><br>
                <span style="color:rgba(255,255,255,0.7)">{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⏱️ Expected Processing Times")
    st.markdown("""
    | Audio Length | Transcription | Summary | Total |
    |---|---|---|---|
    | 1-5 minutes | ~30 sec | ~20 sec | **~1 min** |
    | 5-15 minutes | ~1-2 min | ~30 sec | **~2-3 min** |
    | 15-30 minutes | ~3-5 min | ~1 min | **~5-6 min** |
    | 30+ minutes | 10+ min | ~2 min | **10+ min** |
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-box"><div style="font-size:1.5em">🎵</div><div style="color:#667eea; font-weight:bold;">OpenAI Whisper</div><div class="stat-label">Speech Recognition</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-box"><div style="font-size:1.5em">🤖</div><div style="color:#667eea; font-weight:bold;">Facebook BART</div><div class="stat-label">Text Summarization</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-box"><div style="font-size:1.5em">🌐</div><div style="color:#667eea; font-weight:bold;">Streamlit</div><div class="stat-label">Web Interface</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<br>
<div style="text-align:center; color:rgba(255,255,255,0.4); font-size:0.85em; padding:20px;">
    🎙️ AI Meeting Summarizer | Built with OpenAI Whisper + Facebook BART + Streamlit<br>
    Developed by Koushik K J | Computer Science  &  Engineering
</div>
""", unsafe_allow_html=True)