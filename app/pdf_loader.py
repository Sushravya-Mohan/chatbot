import os
import PyPDF2

def load_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_historical_data():
    """Simulate loading large FAQ/historical data from PDFs."""
    # For simulation, assume you have two PDFs in the "data" folder.
    pdf_files = ["data/faq_data.pdf", "data/historical_conversations.pdf"]
    full_text = ""
    for pdf in pdf_files:
        try:
            full_text += load_pdf_text(pdf)
        except Exception as e:
            print(f"Error loading {pdf}: {e}")
    return full_text
