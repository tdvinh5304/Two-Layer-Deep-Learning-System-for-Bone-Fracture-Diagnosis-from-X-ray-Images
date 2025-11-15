"""
Bone Fracture Detection System - Main Application
Streamlit web application for X-ray fracture analysis
"""

import streamlit as st
from PIL import Image
import sys
from pathlib import Path
import io
import requests
import mysql.connector

# ==============================
# MYSQL CONFIG + SAMPLE FUNCTIONS
# ==============================

MYSQL_HOST = "bf2y1sghovmjlpuyvhxi-mysql.services.clever-cloud.com"
MYSQL_DB   = "bf2y1sghovmjlpuyvhxi"
MYSQL_USER = "un87fm6fqztdwuck"
MYSQL_PORT = 3306
MYSQL_PASS = "QFpD3gRbkukuFak11A67"


def get_all_samples():
    """Lấy (image_path, label) từ MySQL."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS,
            database=MYSQL_DB, port=MYSQL_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT image_path, label FROM mura_images")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return rows

    except Exception as e:
        print("MySQL error:", e)
        return []


def get_dataset_stats():
    """Lấy thống kê chi tiết dataset trong MySQL."""
    conn = mysql.connector.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS,
        database=MYSQL_DB, port=MYSQL_PORT
    )
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) FROM mura_images")
    stats["total_images"] = cursor.fetchone()[0]

    cursor.execute("SELECT dataset, COUNT(*) FROM mura_images GROUP BY dataset")
    stats["dataset_counts"] = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("SELECT label, COUNT(*) FROM mura_images GROUP BY label")
    stats["label_counts"] = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT 
            CASE
                WHEN image_path LIKE '%ELBOW%' THEN 'ELBOW'
                WHEN image_path LIKE '%FINGER%' THEN 'FINGER'
                WHEN image_path LIKE '%FOREARM%' THEN 'FOREARM'
                WHEN image_path LIKE '%HAND%' THEN 'HAND'
                WHEN image_path LIKE '%HUMERUS%' THEN 'HUMERUS'
                WHEN image_path LIKE '%SHOULDER%' THEN 'SHOULDER'
                WHEN image_path LIKE '%WRIST%' THEN 'WRIST'
                ELSE 'UNKNOWN'
            END as bone,
            COUNT(*)
        FROM mura_images
        GROUP BY bone
    """)
    stats["bone_counts"] = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.close()
    conn.close()
    return stats


# ==============================
# SHORT NAME FOR DROPDOWN
# ==============================
def shorten_path(full_path: str, label: int):
    parts = full_path.split("/")
    bone = parts[2] if len(parts) > 2 else "Unknown"
    patient = parts[3] if len(parts) > 3 else "Unknown"
    filename = parts[-1]

    label_text = "Fracture" if label == 1 else "Normal"

    return f"{bone.replace('XR_', '')} | {patient} | {filename} | {label_text}"


# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.model_handler import ModelHandler
from src.predictor import FracturePrediction
from src.visualizer import Visualizer

# >>> ADDED FOR PAGE TAB <<<
from page import run_page


# Page configuration
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CSS
# ==============================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .info-card {
        background-color: #f1f5f9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #6366f1;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# SESSION STATE
# ==============================
def init_session_state():
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False



# ==============================
# LOAD MODELS
# ==============================
def load_models():
    try:
        with st.spinner("🔄 Loading models..."):
            handler = ModelHandler(models_dir="models")
            handler.load_models()
            st.session_state.model_handler = handler
            st.session_state.predictor = FracturePrediction(handler)
            st.session_state.models_loaded = True
            st.success("✅ Models loaded!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")


# ==============================
# MAIN APP
# ==============================
def main():

    init_session_state()

    st.markdown('<h1 class="main-header">🦴 Bone Fracture Detection System</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Control Panel")

        st.markdown("### Model Management")
        if not st.session_state.models_loaded:
            if st.button("Load Models"):
                load_models()
        else:
            st.success("Models Ready")
            if st.button("Reload Models"):
                st.session_state.models_loaded = False
                load_models()

    # >>> ADDED FOR PAGE TAB <<<
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analysis", "📊 Information", "ℹ️ Help", "Assistant Page"])


    # ===========================================================
    # TAB 1: ANALYSIS
    # ===========================================================
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📤 Upload X-ray Image")

            uploaded_file = st.file_uploader(
                "Choose image",
                type=['png', 'jpg', 'jpeg'],
                label_visibility="collapsed"
            )

            if uploaded_file is None:

                st.markdown("### Or select sample")

                sample_rows = get_all_samples()
                short_list = []
                mapping = {}

                for path, label in sample_rows:
                    short_name = shorten_path(path, label)
                    short_list.append(short_name)
                    mapping[short_name] = (path, label)

                selected_short = st.selectbox("Select sample", ["-- select --"] + short_list)

                if selected_short != "-- select --":
                    full_path, lbl = mapping[selected_short]

                    try:
                        if full_path.startswith("http"):
                            r = requests.get(full_path)
                            img = Image.open(io.BytesIO(r.content))
                        else:
                            img = Image.open(full_path)

                        st.session_state.current_image = img
                        st.image(img, caption=f"{selected_short}", use_container_width=True)

                        st.success(f"Label: {'Fracture' if lbl == 1 else 'Normal'}")

                    except Exception as e:
                        st.error(f"❌ Failed: {e}")

            if uploaded_file:
                img = Image.open(uploaded_file)
                st.session_state.current_image = img
                st.image(img, caption=uploaded_file.name, use_container_width=True)


        with col2:
            st.markdown("### 📋 Analysis Results")

            if st.session_state.current_image is not None:
                if st.button("🔬 Analyze Image"):
                    if not st.session_state.models_loaded:
                        st.warning("Load models first!")
                    else:
                        with st.spinner("Analyzing..."):
                            st.session_state.results = st.session_state.predictor.analyze_image(
                                st.session_state.current_image
                            )

                if st.session_state.results:
                    Visualizer.display_results(st.session_state.results)
            else:
                st.info("Upload or select sample first.")



    # ===========================================================
    # TAB 2: INFORMATION
    # ===========================================================
    with tab2:
        st.markdown("## 📊 Dataset Information")

        stats = get_dataset_stats()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Images", stats["total_images"])
        c2.metric("Fracture", stats["label_counts"].get(1, 0))
        c3.metric("Normal", stats["label_counts"].get(0, 0))

        st.markdown("---")
        st.markdown("### 🗂 Dataset Breakdown")
        for k, v in stats["dataset_counts"].items():
            st.write(f"- **{k}**: {v}")

        st.markdown("### 🦴 Bone Type Counts")
        cols = st.columns(4)
        for i, (bone, cnt) in enumerate(stats["bone_counts"].items()):
            cols[i % 4].write(f"**{bone}**: {cnt}")

        st.markdown("---")
        st.markdown("### 🖼 Sample Preview")

        preview_rows = get_all_samples()[:8]
        cols2 = st.columns(4)
        for i, (path, label) in enumerate(preview_rows):
            try:
                if path.startswith("http"):
                    res = requests.get(path)
                    img = Image.open(io.BytesIO(res.content))
                else:
                    img = Image.open(path)

                cols2[i % 4].image(img, caption=shorten_path(path, label), use_container_width=True)
            except:
                pass

        st.markdown("---")
        st.markdown("### 🤖 Model Architecture")

        st.write("""
**Bone Classifier (ResNet50)**  
• Input: 224×224 grayscale  
• Output: 7 bone types  

**Fracture Models (7 models)**  
• Architectures: ResNet50 / InceptionV3  
• Output: fracture probability  
""")


    # ===========================================================
    # TAB 3: HELP
    # ===========================================================
    with tab3:
        st.markdown("## ℹ️ Help & Documentation")

        st.markdown("### 🚀 Quick Start")
        st.write("""
1. Load models  
2. Upload image OR select sample  
3. Click Analyze  
4. View results  
""")

        st.markdown("### 📁 Database Schema")
        st.code("""
mura_images:
  id INT PK
  image_path VARCHAR
  study_path VARCHAR
  label INT (0 normal, 1 fracture)
  dataset VARCHAR
""")

        st.markdown("### 🧠 Pipeline")
        st.write("""
- Preprocessing  
- Bone classification  
- Bone-specific fracture detection  
- Result visualization  
""")

        st.markdown("### 🔧 Troubleshooting")
        st.write("""
- Models not loaded → click Load Models  
- Missing image → check file path  
- Prediction fails → reload models  
""")


    # ===========================================================
    # TAB 4: PAGE.PY
    # ===========================================================
    with tab4:
        run_page()


# RUN
if __name__ == "__main__":
    main()