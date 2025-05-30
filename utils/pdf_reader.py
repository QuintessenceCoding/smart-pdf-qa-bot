# pdf_reader.py

import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    """
    Extracts text from every page of a PDF file.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        str: Combined text from all pages
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
