from PIL import Image, ImageDraw, ImageFont
import sys
from pathlib import Path

img = Image.new("RGB", (768, 1024), (20, 20, 20))
draw = ImageDraw.Draw(img)
font_path = Path("fadegoblin_playwright/src/fadegoblin/assets/Inter.ttf")
font = ImageFont.truetype(str(font_path), 80)
font_sub = ImageFont.truetype(str(font_path), 40)

text1 = "CHC @ PIT"
text2 = "+146 (+140)"
text3 = "+9.3% EV • ⭐⭐⭐"

y = 700

for text, f, color in [(text1, font, "white"), (text2, font, (255, 255, 100)), (text3, font_sub, (200, 255, 200))]:
    bbox = draw.textbbox((0, 0), text, font=f)
    w = bbox[2] - bbox[0]
    x = (768 - w) / 2
    
    # Outer glow (cyan)
    draw.text((x, y), text, font=f, fill=(0, 255, 255), stroke_width=12, stroke_fill=(0, 255, 255))
    # Inner stroke (black)
    draw.text((x, y), text, font=f, fill=color, stroke_width=6, stroke_fill="black")
    
    y += (bbox[3] - bbox[1]) + 20

img.save("scratch/graffiti_test.png")
print("Done")
