# frontend/app.py
import streamlit as st
from pathlib import Path
import uuid

# ----------------------------
# Config
# ----------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # create folder if it doesn't exist

MAX_FILE_SIZE_MB = 5
ALLOWED_EXT = {".pdf", ".docx"}

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(
    page_title="SmartApply",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📄 SmartApply — Resume Upload")
st.markdown(
    """
    **Instructions:** Upload your resume in PDF or DOCX format.
    Maximum file size: 5 MB.
    """
)

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Drag & drop or click to upload",
    type=[ext.strip('.') for ext in ALLOWED_EXT]
)

# ----------------------------
# Handle File
# ----------------------------
if uploaded_file is not None:
    # Check file size
    size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large: {size_mb:.2f} MB (max {MAX_FILE_SIZE_MB} MB).")
    else:
        # Generate unique filename to avoid collisions
        unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        file_path = UPLOAD_DIR / unique_name

        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Show success message
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.write(f"📂 Saved to: {file_path}")
        st.write(f"📏 Size: {size_mb:.2f} MB")

# ----------------------------
# Optional: show recent uploads
# ----------------------------
recent_files = list(UPLOAD_DIR.glob("*"))
if recent_files:
    st.markdown("#### Recent uploads (last 5)")
    for f in sorted(recent_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
        st.write(f"- `{f.name}` ({round(f.stat().st_size/(1024*1024), 2)} MB)")
