"""
Text Normalization & Preprocessing Module
Cleans whitespace, standardizes unicode, and prepares benchmark text for rendering.
"""

import re
import unicodedata

def normalize_text(text: str) -> str:
    """
    Standardize text formatting for rasterization:
    - Normalizes unicode characters (NFKC)
    - Normalizes line breaks
    - Cleans irregular spaces
    """
    if not text:
        return ""
    
    # Unicode standard normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Normalize Windows/Unix line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Clean multiple consecutive horizontal spaces (preserving newlines)
    text = re.sub(r"[ \t]+", " ", text)
    
    # Strip leading/trailing outer whitespace
    return text.strip()
