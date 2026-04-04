
import streamlit as st
import requests

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and analyze it with ATS + AI Q&A")

# -------------------------------
# Upload Resume
# -------------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# -------------------------------
# Job Description Input
# -------------------------------
jd = st.text_area("Enter Job Description")

# -------------------------------
# ANALYZE SECTION
# -------------------------------
if st.button("Analyze Resume"):
    if uploaded_file and jd:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/analyze",
                files={"file": uploaded_file},
                data={"job_description": jd}
            )

            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("📊 ATS Score")
                st.success(result.get("ATS_score", "Not available"))

                st.subheader("✅ Matched Skills")
                st.write(result.get("matched_skills", []))

                st.subheader("❌ Missing Skills")
                st.write(result.get("missing_skills", []))

                # ✅ Debug safely here
                st.write("DEBUG:", result)

        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
            
# -------------------------------
# RAG Q&A SECTION
# -------------------------------
st.markdown("---")
st.subheader("🤖 Ask Questions about Resume")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if uploaded_file and question:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                files={"file": uploaded_file},
                data={"question": question}
            )

            result = response.json()

            # 🛑 Handle backend errors
            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("💡 Answer")
                st.success(result.get("answer", "No response"))

        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

    else:
        st.warning("Upload resume and enter a question")

# -------------------------------
# DEBUG (optional)
# -------------------------------
if 'result' in locals():
