# frontend/app.py
from __future__ import annotations

import os
import json
import uuid
import math
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# =========================
# Bootstrap & Config
# =========================
load_dotenv()  # Loads variables from .env at project root

# Keys & endpoints
AFFINDA_API_KEY = os.getenv("AFFINDA_API_KEY", "")

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "ca")

CORESIGNAL_API_KEY = os.getenv("CORESIGNAL_API_KEY", "")
CORESIGNAL_BASE_URL = os.getenv("CORESIGNAL_BASE_URL", "https://api.coresignal.com/jobs")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-4o-mini")

# Local upload policy (preserves your original UX)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE_MB = 5
ALLOWED_EXT = {".pdf", ".docx"}

# =========================
# Utility Helpers
# =========================
def _have_affinda() -> bool: return bool(AFFINDA_API_KEY)
def _have_adzuna() -> bool: return bool(ADZUNA_APP_ID and ADZUNA_APP_KEY)
def _have_coresignal() -> bool: return bool(CORESIGNAL_API_KEY)
def _have_openai() -> bool: return bool(OPENAI_API_KEY)

def _bytes_to_mb(nbytes: int) -> float:
    return nbytes / (1024 * 1024)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# =========================
# 1) Résumé Parsing — Affinda
# =========================
def parse_resume_with_affinda(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Affinda documents:analyze endpoint.
    Docs: https://docs.affinda.com/
    """
    if not _have_affinda():
        return {"error": "AFFINDA_API_KEY missing in environment."}

    url = "https://api.affinda.com/v3/documents:analyze"
    headers = {"Authorization": f"Bearer {AFFINDA_API_KEY}"}
    files = {"file": (filename, file_bytes)}
    data = {"wait": "true"}  # wait until parsing completes

    try:
        r = requests.post(url, headers=headers, files=files, data=data, timeout=90)
    except Exception as e:
        return {"error": f"Affinda request failed: {e}"}

    if r.status_code >= 400:
        return {"error": f"Affinda error {r.status_code}: {r.text}"}
    return r.json()

def summarize_candidate(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Schema-tolerant extract of candidate profile from Affinda JSON.
    """
    out = {
        "name": None,
        "emails": [],
        "phones": [],
        "skills": [],
        "education": [],
        "experiences": [],
    }
    try:
        data = parsed.get("data") or {}
        out["name"] = (data.get("name") or {}).get("raw") or data.get("name")
        out["emails"] = data.get("emails") or []
        out["phones"] = data.get("phoneNumbers") or []

        # skills: list of {"name": "..."}
        out["skills"] = [s.get("name") for s in (data.get("skills") or []) if s.get("name")]

        # education
        for edu in (data.get("education") or []):
            out["education"].append({
                "organization": edu.get("organization"),
                "accreditation": (edu.get("accreditation") or {}).get("educationLevel"),
                "grade": edu.get("grade"),
                "startDate": (edu.get("dates") or {}).get("startDate"),
                "endDate": (edu.get("dates") or {}).get("endDate"),
            })

        # work experience
        for exp in (data.get("workExperience") or []):
            out["experiences"].append({
                "jobTitle": exp.get("jobTitle"),
                "organization": exp.get("organization"),
                "location": exp.get("location"),
                "startDate": (exp.get("dates") or {}).get("startDate"),
                "endDate": (exp.get("dates") or {}).get("endDate"),
                "summary": exp.get("jobDescription"),
            })
    except Exception as e:
        out["parse_warning"] = f"Partial parse warning: {e}"
    return out

# =========================
# 2) Job Search — Adzuna
# =========================
def adzuna_search_jobs(query: str, location: str = "London, ON", results: int = 20) -> List[Dict[str, Any]]:
    """
    Adzuna search API (pages of up to 20).
    Docs: https://developer.adzuna.com/docs/search
    """
    if not _have_adzuna():
        return [{"error": "ADZUNA_APP_ID / ADZUNA_APP_KEY missing in environment."}]

    per_page = min(20, results)
    pages = math.ceil(results / per_page)
    jobs: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/{page}"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "what": query,
            "where": location,
            "results_per_page": per_page,
            "content-type": "application/json",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
        except Exception as e:
            jobs.append({"error": f"Adzuna request failed: {e}"})
            break

        if r.status_code >= 400:
            jobs.append({"error": f"Adzuna error {r.status_code}: {r.text}"})
            break

        payload = r.json()
        jobs.extend(payload.get("results", []))
        if len(jobs) >= results:
            break

    return jobs[:results]

# =========================
# 2b) Job Search — CoreSignal
# =========================
def coresignal_search_jobs(query: str, location: str = "Canada", limit: int = 10) -> List[Dict[str, Any]]:
    """
    CoreSignal Jobs API.
    Some accounts use /jobs/search, others /v1/jobs/search. Configure CORESIGNAL_BASE_URL in .env if needed.
    """
    if not _have_coresignal():
        return [{"error": "CORESIGNAL_API_KEY missing in environment."}]

    base = CORESIGNAL_BASE_URL.rstrip("/")
    # If base ends with /jobs, append /search
    url = base + ("/search" if base.endswith("/jobs") else "")

    headers = {
        "Authorization": f"Bearer {CORESIGNAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "search": query,
        "location": location,
        "size": limit
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        return [{"error": f"CoreSignal request failed: {e}"}]

    if r.status_code >= 400:
        return [{"error": f"CoreSignal error {r.status_code}: {r.text}"}]

    data = r.json()
    # Normalize to list of jobs
    if isinstance(data, dict):
        if isinstance(data.get("jobs"), list):
            return data["jobs"]
        if isinstance(data.get("results"), list):
            return data["results"]
        # Some responses may be a dict with a single "data" list
        if isinstance(data.get("data"), list):
            return data["data"]
        return [data]
    return data if isinstance(data, list) else [data]

# =========================
# 3) OpenAI — Embeddings Ranking & Cover Letter (optional)
# =========================
def _get_openai_embedding(text: str) -> Optional[np.ndarray]:
    if not _have_openai():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(input=text, model=OPENAI_EMBEDDING_MODEL)
        vec = resp.data[0].embedding
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return None

def _candidate_text(profile: Dict[str, Any]) -> str:
    parts = []
    if profile.get("skills"):
        parts.append("Skills: " + ", ".join(profile["skills"]))
    if profile.get("experiences"):
        role_bits = []
        for e in profile["experiences"][:6]:
            t = f"{(e.get('jobTitle') or '').strip()} at {(e.get('organization') or '').strip()}: {(e.get('summary') or '').strip()}"
            role_bits.append(t.strip())
        if role_bits:
            parts.append("Experience: " + " | ".join(role_bits))
    if profile.get("education"):
        edu_bits = []
        for e in profile["education"][:4]:
            edu_bits.append(f"{(e.get('accreditation') or '').strip()} at {(e.get('organization') or '').strip()}")
        if edu_bits:
            parts.append("Education: " + " | ".join(edu_bits))
    return "\n".join(parts).strip()

def rank_jobs_by_similarity(candidate_profile: Dict[str, Any],
                            jobs: List[Dict[str, Any]],
                            top_k: int = 5) -> List[Dict[str, Any]]:
    cand_text = _candidate_text(candidate_profile)
    cand_vec = _get_openai_embedding(cand_text) if cand_text else None
    if cand_vec is None:
        return jobs[:top_k]

    scored = []
    for j in jobs:
        # Build comparable text from various job schemas
        title = j.get("title") or ""
        company = ""
        if isinstance(j.get("company"), dict):
            company = j["company"].get("display_name") or j["company"].get("name") or ""
        elif isinstance(j.get("company"), str):
            company = j["company"]
        desc = j.get("description") or j.get("job_description") or j.get("description_text") or ""
        job_text = f"{title} at {company}\n{desc}".strip()

        job_vec = _get_openai_embedding(job_text)
        sim = 0.0 if job_vec is None else _cosine_sim(cand_vec, job_vec)

        j_copy = dict(j)
        j_copy["_similarity"] = sim
        scored.append(j_copy)

    scored.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
    return scored[:top_k]

def generate_cover_letter(profile: Dict[str, Any], job: Dict[str, Any]) -> str:
    if not _have_openai():
        return "OpenAI key not set. Add OPENAI_API_KEY to .env to enable cover-letter generation."

    name = profile.get("name") or "Candidate"
    skills = ", ".join(profile.get("skills", [])[:20]) or "relevant skills"

    title = job.get("title") or "the role"
    if isinstance(job.get("company"), dict):
        company = job["company"].get("display_name") or job["company"].get("name") or "your company"
    else:
        company = job.get("company") or "your company"
    description = job.get("description") or job.get("job_description") or ""

    prompt = f"""
You are a concise Canadian career writer. Write a tailored cover letter (max 300 words).
Role: {title}
Company: {company}
Location context: London, Ontario, Canada.

Candidate: {name}
Skills: {skills}
Top experiences: {json.dumps(profile.get('experiences', [])[:3], ensure_ascii=False)}

Map skills to the job description, use Canadian spelling, and end with a confident call-to-action.
Avoid clichés and filler. Keep paragraphs tight.
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_GPT_MODEL,
            messages=[
                {"role": "system", "content": "You write succinct, high-impact cover letters for Canadian job applications."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Cover-letter generation error: {e}"

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SmartApply", page_icon="📝", layout="centered", initial_sidebar_state="expanded")
st.title("📄 SmartApply — Upload → Parse → Jobs → Match → Cover Letter")
st.caption("Keys are read from .env. Never hard-code secrets.")

with st.sidebar:
    st.subheader("🔐 API Status")
    st.write(f"Affinda: {'✅' if _have_affinda() else '❌'}")
    st.write(f"Adzuna: {'✅' if _have_adzuna() else '❌'}")
    st.write(f"CoreSignal: {'✅' if _have_coresignal() else '❌'}")
    st.write(f"OpenAI: {'✅' if _have_openai() else '❌'}")
    st.caption("Set values in .env at project root.")

# --- 1) Upload & Parse ---
st.markdown("### 1) Upload your résumé (PDF or DOCX)")
uploaded = st.file_uploader(
    "Drag & drop or click to upload",
    type=[ext.strip('.') for ext in ALLOWED_EXT]
)

candidate_profile: Optional[Dict[str, Any]] = None
if uploaded is not None:
    size_mb = _bytes_to_mb(len(uploaded.getbuffer()))
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large: {size_mb:.2f} MB (max {MAX_FILE_SIZE_MB} MB).")
    else:
        unique_name = f"{uuid.uuid4().hex}_{uploaded.name}"
        path = UPLOAD_DIR / unique_name
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Uploaded: {uploaded.name}")
        st.write(f"📂 Saved to: {path}")
        st.write(f"📏 Size: {size_mb:.2f} MB")

        with st.spinner("Parsing résumé with Affinda..."):
            parsed = parse_resume_with_affinda(uploaded.getbuffer(), uploaded.name)
        if "error" in parsed:
            st.error(parsed["error"])
        else:
            candidate_profile = summarize_candidate(parsed)
            st.markdown("#### Parsed Candidate Summary")
            st.json(candidate_profile)

# Recent uploads
recent_files = sorted(UPLOAD_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
if recent_files:
    st.markdown("#### Recent uploads (last 5)")
    for f in recent_files:
        st.write(f"- `{f.name}` ({round(f.stat().st_size/(1024*1024), 2)} MB)")

st.markdown("---")

# --- 2) Job Search ---
st.markdown("### 2) Search jobs")
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    job_query = st.text_input("Keywords", value="data analyst")
with c2:
    job_location = st.text_input("Location", value="London, ON")
with c3:
    job_limit = st.number_input("Results", value=10, min_value=1, max_value=50, step=1)

source = st.radio("Job Source", ["Adzuna", "CoreSignal"], horizontal=True)

jobs: List[Dict[str, Any]] = []
if st.button("🔎 Search Jobs"):
    with st.spinner(f"Fetching jobs from {source}..."):
        if source == "Adzuna":
            jobs = adzuna_search_jobs(job_query, job_location, job_limit)
        else:
            # CoreSignal locations often expect a country/region string; keep it flexible
            cs_location = job_location or "Canada"
            jobs = coresignal_search_jobs(job_query, cs_location, job_limit)

    if not jobs:
        st.info("No jobs found.")
    else:
        for j in jobs:
            if isinstance(j, dict) and "error" in j:
                st.error(j["error"])
                continue

            # Normalize display fields
            title = j.get("title") or "Untitled"
            company = ""
            if isinstance(j.get("company"), dict):
                company = j["company"].get("display_name") or j["company"].get("name") or "Unknown"
            elif isinstance(j.get("company"), str):
                company = j["company"] or "Unknown"
            else:
                company = "Unknown"

            # Location display
            location_str = ""
            loc = j.get("location")
            if isinstance(loc, dict):
                # Adzuna: {"display_name": "...", "area": [...]}
                location_str = loc.get("display_name") or ""
                if not location_str and isinstance(loc.get("area"), list) and loc["area"]:
                    location_str = ", ".join(loc["area"])
            elif isinstance(loc, str):
                location_str = loc

            st.markdown(f"**{title}** — *{company}*")
            if location_str:
                st.caption(location_str)

            desc = j.get("description") or j.get("job_description") or j.get("description_text") or ""
            if desc:
                with st.expander("Job description"):
                    st.write(desc)

            # External apply / source URL
            redir = j.get("redirect_url") or j.get("apply_url") or j.get("url")
            if redir:
                st.write(f"[Apply / View Posting]({redir})")
            st.divider()

st.markdown("---")

# --- 3) Ranking & Cover Letter ---
st.markdown("### 3) Match & generate tailored cover letter (optional OpenAI)")
top_k = st.slider("How many top matches to keep?", 1, 10, 5, 1)

if st.button("🎯 Rank Jobs & ✍️ Generate Cover Letter"):
    if not jobs:
        st.warning("Run a job search first.")
    elif not candidate_profile:
        st.warning("Upload and parse a résumé first.")
    else:
        ranked = jobs
        if _have_openai():
            with st.spinner("Ranking jobs by semantic similarity (OpenAI embeddings)..."):
                ranked = rank_jobs_by_similarity(candidate_profile, jobs, top_k=top_k)
        else:
            ranked = jobs[:top_k]

        if not ranked:
            st.info("No ranked jobs available.")
        else:
            st.success("Top matches ready. Expand a job to generate a tailored cover letter.")
            for i, job in enumerate(ranked, start=1):
                sim = job.get("_similarity")
                label = f"{i}. {job.get('title') or 'Untitled'}"
                if isinstance(sim, (int, float)):
                    label += f" — Similarity: {sim:.3f}"

                with st.expander(label):
                    comp = ""
                    if isinstance(job.get("company"), dict):
                        comp = job["company"].get("display_name") or job["company"].get("name") or "Unknown"
                    elif isinstance(job.get("company"), str):
                        comp = job["company"]
                    comp = comp or "Unknown"
                    st.write(f"**Company:** {comp}")

                    jdesc = job.get("description") or job.get("job_description") or ""
                    if jdesc:
                        st.markdown("**Description:**")
                        st.write(jdesc)

                    if st.button(f"✉️ Generate Cover Letter for Match #{i}", key=f"cl_{i}"):
                        if not _have_openai():
                            st.warning("OpenAI key not set. Add OPENAI_API_KEY in .env to enable this.")
                        else:
                            with st.spinner("Composing tailored cover letter..."):
                                letter = generate_cover_letter(candidate_profile, job)
                            st.markdown("#### Tailored Cover Letter")
                            st.write(letter)
