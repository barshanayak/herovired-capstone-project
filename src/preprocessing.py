import re
from PyPDF2 import PdfReader


def clean_text(text: str) -> str:
    """Normalize whitespace and return a clean single-line string."""
    text = str(text)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf_resume(uploaded_file) -> str:
    """Extract readable text from a PDF resume file-like object."""
    try:
        reader = PdfReader(uploaded_file)
        text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        return clean_text(" ".join(text))

    except Exception:
        return ""


def extract_jd_resume(text: str):
    text = str(text)

    jd_match = re.search(
        r"For the given job description\s*<<(.*?)>>\s*the resume:",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    resume_match = re.search(
        r"the resume:\s*<<(.*?)>>\.?\s*The result is",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    jd = jd_match.group(1).strip() if jd_match else ""
    resume = resume_match.group(1).strip() if resume_match else ""

    return jd, resume