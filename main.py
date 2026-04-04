
from fastapi import FastAPI, File, UploadFile, Form
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import tempfile

# RAG imports
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import hashlib

app = FastAPI()

# -------------------------------
# INIT MODEL
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# CACHE (IMPORTANT)
# -------------------------------
cached_index = None
cached_chunks = None
cached_resume_hash = None

# -------------------------------
# SKILLS (IMPROVED)
# -------------------------------
SKILLS = {
    "python": ["python"],
    "sql": ["sql"],
    "machine learning": ["machine learning", "ml"],
    "data analysis": ["data analysis", "data analytics"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language processing"],
    "fastapi": ["fastapi"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "statistics": ["statistics", "statistical"]
}

# -------------------------------
# PDF EXTRACTION
# -------------------------------
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    return text

# -------------------------------
# ATS SCORE
# -------------------------------
def compute_ats_score(resume_text, jd_text):
    documents = [resume_text, jd_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = similarity[0][0] * 100

    return round(score, 2)

# -------------------------------
# SKILL MATCHING
# -------------------------------
def skill_match(resume_text, jd_text):
    resume_text = resume_text.lower()
    jd_text = jd_text.lower()

    matched = []
    missing = []

    for skill, variations in SKILLS.items():
        if any(var in jd_text for var in variations):
            if any(var in resume_text for var in variations):
                matched.append(skill)
            else:
                missing.append(skill)

    return matched, missing

# -------------------------------
# CHUNKING (SMALLER)
# -------------------------------
def split_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# -------------------------------
# VECTOR STORE
# -------------------------------
def create_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks

# -------------------------------
# RETRIEVE (FAST)
# -------------------------------
def retrieve(query, index, chunks, k=1):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    return [chunks[i] for i in indices[0] if i < len(chunks)]

# -------------------------------
# FAST OLLAMA CALL
# -------------------------------
def generate_answer(context, question):
    prompt = f"""
    Answer briefly in 2-3 lines.

    Context:
    {context}

    Question:
    {question}
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b-instruct-q4_0",  # 🔥 FAST MODEL
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 100  # 🔥 LIMIT OUTPUT
            }
        }
    )

    return response.json()["response"]

# -------------------------------
# API: ATS
# -------------------------------
@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    resume_text = extract_text_from_pdf(temp_path)

    if not resume_text.strip():
        return {"error": "Could not extract text from PDF"}

    score = compute_ats_score(resume_text, job_description)
    matched, missing = skill_match(resume_text, job_description)

    return {
        "ATS_score": score,
        "matched_skills": matched,
        "missing_skills": missing
    }

# -------------------------------
# API: RAG (OPTIMIZED)
# -------------------------------
@app.post("/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    global cached_index, cached_chunks, cached_resume_hash

    file_bytes = await file.read()
    resume_hash = hashlib.md5(file_bytes).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(file_bytes)
        temp_path = temp.name

    resume_text = extract_text_from_pdf(temp_path)

    if not resume_text.strip():
        return {"error": "Could not extract text from PDF"}

    # ✅ Cache logic
    if cached_resume_hash != resume_hash:
        chunks = split_text(resume_text)
        index, chunks = create_vector_store(chunks)

        cached_index = index
        cached_chunks = chunks
        cached_resume_hash = resume_hash

    # Retrieval
    relevant_chunks = retrieve(question, cached_index, cached_chunks)
    context = " ".join(relevant_chunks)

    answer = generate_answer(context, question)
