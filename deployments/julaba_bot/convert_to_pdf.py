#!/usr/bin/env python3
"""Convert markdown to PDF using WeasyPrint"""
from pathlib import Path
from markdown2 import markdown

try:
    from weasyprint import HTML
    weasyprint_available = True
except ImportError:
    weasyprint_available = False

workspace = Path("/home/opc/julaba")
md_file = workspace / "JULABA_COMPLETE_DOCUMENTATION.md"
pdf_file = workspace / "JULABA_COMPLETE_DOCUMENTATION.pdf"
html_file = workspace / "JULABA_COMPLETE_DOCUMENTATION.html"

print(f"📄 Converting {md_file.name} to PDF...")

# Read markdown
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert to HTML
print(f"   Converting markdown to HTML...")
html_content = markdown(md_content, extras=['tables', 'fenced-code-blocks'])

# Wrap in proper HTML
full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>JULABA Complete Documentation</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 2cm;
            font-size: 11pt;
        }}
        h1 {{ color: #1a5490; border-bottom: 3px solid #1a5490; padding-bottom: 10px; }}
        h2 {{ color: #2d7ab8; margin-top: 30px; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }}
        pre {{ background-color: #f4f4f4; padding: 12px; border-left: 4px solid #2d7ab8; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        table th {{ background-color: #2d7ab8; color: white; padding: 10px; }}
        table td {{ border: 1px solid #ddd; padding: 8px; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Save HTML
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(full_html)
print(f"   ✅ HTML created")

# Convert to PDF
if weasyprint_available:
    try:
        print(f"   Converting to PDF...")
        HTML(string=full_html).write_pdf(pdf_file)
        if pdf_file.exists():
            pdf_size = pdf_file.stat().st_size / 1024 / 1024
            print(f"\n✅ PDF CREATED!")
            print(f"   File: {pdf_file.name} ({pdf_size:.2f} MB)")
        else:
            print(f"   ❌ PDF creation failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print(f"   ⚠️ WeasyPrint not available")
