# ------------------------------------------SmartApplyAI------------------------------------------

import os, io, uuid, sqlite3
from pathlib import Path
import streamlit as st
from passlib.hash import pbkdf2_sha256
import fitz  # PyMuPDF for PDF parsing
from docx import Document  # python-docx for DOCX
import spacy
from sentence_transformers import SentenceTransformer
import requests

# --------------API KEYS--------------
AFFINDA_API_KEY = "aff_e54673a5b5911273e9e6c3406c37ac61609e2a3e"
ADZUNA_APP_ID = "65243338"
ADZUNA_APP_KEY = "afa8a465c768291a4e8096e3e83e839c"
ADZUNA_COUNTRY = "ca"

# --------------Database and Folders--------------
DB_PATH = "smartapply.db"
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          email TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          filename TEXT NOT NULL,
          storage_path TEXT NOT NULL,
          full_text TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)
init_db()

# --------------User Authorization--------------
def signup(email, password):
    ph = pbkdf2_sha256.hash(password)
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO users(email, password_hash) VALUES (?, ?)", (email, ph))
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Email already exists."

def login(email, password):
    with get_conn() as conn:
        row = conn.execute("SELECT id, password_hash FROM users WHERE email=?", (email,)).fetchone()
    if row and pbkdf2_sha256.verify(password, row["password_hash"]):
        return row["id"]
    return None

def current_user():
    return st.session_state.get("user")

def require_auth():
    user = current_user()
    if not user:
        st.warning("Please log in to continue.")
        st.stop()
    return user

def logout():
    st.session_state.pop("user", None)

# --------------Resume Parsing--------------
def extract_text_from_pdf(bytes_data):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(stream=bytes_data, filetype="pdf")
    text = ""
    for page in doc:
        t = page.get_text("text")
        if t:
            text += t + "\n"
    return text.strip()

def extract_text_from_docx(bytes_data):
    """Extracts text from a DOCX file using python-docx."""
    doc = Document(io.BytesIO(bytes_data))
    return "\n".join(p.text for p in doc.paragraphs)

def handle_upload(user_id, file):
    """Handles file upload and stores extracted text in DB."""
    name = file.name
    ext = name.split(".")[-1].lower()
    data = file.read()
    uid = f"{uuid.uuid4()}.{ext}"
    save_path = UPLOAD_DIR / uid
    save_path.write_bytes(data)
    if ext == "pdf":
        text = extract_text_from_pdf(data)
    elif ext == "docx":
        text = extract_text_from_docx(data)
    else:
        raise ValueError("Unsupported file type. Please upload PDF or DOCX.")
    # Save to DB
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO uploads(user_id, filename, storage_path, full_text)
            VALUES (?, ?, ?, ?)""",
            (user_id, name, str(save_path), text)
        )
    return text, save_path, name, data

# --------------NLP Model Load--------------
@st.cache_resource
def load_nlp_models():
    nlp = spacy.load("en_core_web_lg")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, sbert
NLP, SBERT = load_nlp_models()

# --------------Adzuna Job Search--------------
def fetch_jobs_adzuna(query, location="Toronto", results=10, days=3):
    """Fetch jobs from Adzuna API."""
    url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": location,
        "results_per_page": results,
        "max_days_old": days,
        "sort_by": "date",
        "content-type": "application/json"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        st.error(f"Job fetch error: {e}")
        return []

# --------------User Interface--------------
st.title("SmartApply: AI Job Application Helper")

with st.sidebar:
    page = st.radio("Navigate", ["Login / Sign up", "Upload Resume", "Parsed Resume Text", "Find Jobs", "Account"])
    if current_user():
        st.caption(f"Logged in as **{current_user()['email']}**")

# --------------Login / Sign up--------------
if page == "Login / Sign up":
    tab1, tab2 = st.tabs(["Login", "Sign up"])
    with tab1:
        login_email = st.text_input("Email")
        login_password = st.text_input("Password", type="password")
        if st.button("Login"):
            user_id = login(login_email, login_password)
            if user_id:
                st.session_state["user"] = {"id": user_id, "email": login_email}
                st.success("Logged in.")
            else:
                st.error("Invalid credentials.")
    with tab2:
        signup_email = st.text_input("Email ")
        signup_password = st.text_input("Password ", type="password")
        if st.button("Create account"):
            signup_ok, signup_msg = signup(signup_email, signup_password)
            if signup_ok:
                st.success(signup_msg)
            else:
                st.error(signup_msg)

# --------------Upload Resume--------------
elif page == "Upload Resume":
    user = require_auth()
    st.subheader("Upload your resume (PDF/DOCX)")
    f = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if f:
        try:
            # Extract and save text, get file bytes for advanced parsing
            full_text, save_path, orig_name, file_bytes = handle_upload(user["id"], f)
            st.success("Uploaded and parsed.")
            st.subheader("Full Extracted Resume Text")
            st.code(full_text, language="text")
        except Exception as e:
            st.error(str(e))

# --------------Parsed Resume Text--------------
elif page == "Parsed Resume Text":
    user = require_auth()
    with get_conn() as conn:
        row = conn.execute("SELECT full_text FROM uploads WHERE user_id=? ORDER BY id DESC LIMIT 1", (user["id"],)).fetchone()
    if not row:
        st.info("Upload a resume first."); st.stop()
    full_text = row["full_text"] or ""
    st.subheader("Final Resume Text Used for Job Search")
    st.code(full_text, language="text")

# --------------Find Jobs--------------
elif page == "Find Jobs":
    user = require_auth()
    st.subheader("Search Jobs (Adzuna)")
    job_query = st.text_input("Job Title or Keywords", value="data analyst")
    job_loc = st.text_input("Location", value="Toronto")
    job_days = st.number_input("Jobs posted in last N days", min_value=1, max_value=30, value=3)
    if st.button("Fetch Jobs"):
        jobs = fetch_jobs_adzuna(job_query, job_loc, results=10, days=job_days)
        if jobs:
            for job in jobs:
                st.markdown(f"**{job.get('title','(no title)')}**  \n{job.get('company',{}).get('display_name','')} — {job.get('location',{}).get('display_name','')}")
                st.caption(job.get("created"))
                st.write(job.get("description",""))
                if job.get("redirect_url"):
                    st.markdown(f"[Apply Here]({job['redirect_url']})")
                st.markdown("---")
        else:
            st.warning("No jobs found or API error.")

# --------------Account--------------
elif page == "Account":
    user = current_user()
    if user:
        st.write(f"Logged in as **{user['email']}**")
        if st.button("Log out"):
            logout()
            st.success("Logged out.")
    else:
        st.info("Not logged in.")
