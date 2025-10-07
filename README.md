# SmartApply – AI Job Application Helper
SmartApply is a web application that helps job seekers securely upload their resumes and discover relevant job postings using AI-powered parsing and the Adzuna Jobs API.
## Key Features
- Secure user authentication (sign up & login with password hashing)
- Upload and parse resumes in PDF or DOCX format
- View parsed resume text directly in the app
- Search for job postings by title, location, and recent postings
- Detailed job listings with company, location, and direct apply links
## Quick Start
1. Clone this repository
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run the app:  
   `streamlit run <your_app_file.py>`
4. Open the browser at the Streamlit local URL
## Project Structure
- `code/` – Main application code
- `data/` – Database and upload files
- `docs/` – Documentation and instructions
- `proposal/` – Project proposal and planning documents
## Main Technologies
- Python 3.8+
- Streamlit
- passlib
- fitz (PyMuPDF)
- python-docx
- requests
- spacy
- sentence-transformers
- sqlite3
## API Keys Required
- Adzuna Jobs API (free account for testing)
- Affinda Resume Parser API (optional for advanced parsing)
## Contributors
AIM1-Group6  
Fanshawe College, Ontario, Canada
