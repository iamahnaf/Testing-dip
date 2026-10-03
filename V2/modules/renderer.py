"""Py mirror of 01_renderer_validation.ipynb (for git diff only; Kaggle runs the .ipynb)."""
import re, textwrap, unicodedata
from PIL import Image, ImageDraw, ImageFont


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "")
    return re.sub(r"[ \t]+", " ", t.replace("\r\n", "\n")).strip()


def render_pil(text: str, width=800, font_size=18, line_spacing=6, wrap=80, pad=30) -> Image.Image:
    text = normalize_text(text)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    lines: list[str] = []
    for para in text.split("\n\n"):
        lines.extend(textwrap.wrap(para, width=wrap))
        lines.append("")
    img = Image.new("RGB", (width, 2 * pad + max(len(lines) * (font_size + line_spacing), 100)), "white")
    d = ImageDraw.Draw(img)
    y = pad
    for ln in lines:
        if ln:
            d.text((pad, y), ln, fill="black", font=font)
        y += font_size + line_spacing
    return img


def bbox_fill_ratio(img: Image.Image, bg_thresh: int = 240) -> float:
    gray = img.convert("L")
    bbox = gray.point(lambda p: 0 if p > bg_thresh else 255, mode="1").getbbox()
    if not bbox:
        return 0.0
    return ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (gray.width * gray.height)
