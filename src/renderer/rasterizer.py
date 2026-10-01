"""
Text-to-Image Rasterizer Module
Implements discrete token budget scaling (70, 140, 280, 560, 1120 visual tokens)
and crisp high-contrast document rendering as specified in the methodology.
"""

import io
import textwrap
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from src.renderer.text_normalizer import normalize_text

# Visual Token Budget target configurations
# Higher token budgets allow larger canvas resolutions and font rendering details
BUDGET_CONFIGS = {
    70:   {"width": 400,  "font_size": 14, "line_spacing": 4, "wrap_width": 55, "padding": 20},
    140:  {"width": 600,  "font_size": 16, "line_spacing": 5, "wrap_width": 70, "padding": 25},
    280:  {"width": 800,  "font_size": 18, "line_spacing": 6, "wrap_width": 80, "padding": 30},
    560:  {"width": 1100, "font_size": 20, "line_spacing": 7, "wrap_width": 95, "padding": 35},
    1120: {"width": 1400, "font_size": 22, "line_spacing": 8, "wrap_width": 115, "padding": 40},
}

class TextRasterizer:
    def __init__(self, default_font_path: Optional[str] = "arial.ttf"):
        self.default_font_path = default_font_path

    def render(
        self,
        text: str,
        token_budget: int = 280,
        bg_color: str = "white",
        text_color: str = "black"
    ) -> Image.Image:
        """
        Renders normalized text into an Image object configured for the target visual token budget.
        """
        text = normalize_text(text)
        config = BUDGET_CONFIGS.get(token_budget, BUDGET_CONFIGS[280])

        font_size = config["font_size"]
        image_width = config["width"]
        padding = config["padding"]
        line_spacing = config["line_spacing"]
        wrap_width = config["wrap_width"]

        # Load font
        try:
            font = ImageFont.truetype(self.default_font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Wrap text by paragraphs
        formatted_lines = []
        for paragraph in text.split("\n\n"):
            wrapped = textwrap.wrap(paragraph, width=wrap_width)
            formatted_lines.extend(wrapped)
            formatted_lines.append("")  # paragraph separator

        line_height = font_size + line_spacing
        image_height = 2 * padding + max(len(formatted_lines) * line_height, 100)

        # Draw onto canvas
        image = Image.new("RGB", (image_width, image_height), color=bg_color)
        draw = ImageDraw.Draw(image)

        y = padding
        for line in formatted_lines:
            if line:
                draw.text((padding, y), line, fill=text_color, font=font)
            y += line_height

        return image

    def to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Converts PIL Image to raw bytes for model API payloads."""
        buf = io.BytesIO()
        image.save(buf, format=format)
        return buf.getvalue()
