#!/usr/bin/env python3
"""
Combine all markdown files into one comprehensive document and convert to PDF
"""
import os
import subprocess
from pathlib import Path

# Define markdown files to include (in order)
md_files = [
    "README.md",
    "SYSTEM_DOCUMENTATION.md",
    "SYSTEM_AUDIT_2026-01-14.md",
    "SYSTEM_ANALYSIS_2026-01-11.md",
    "AI_SYSTEM_VERIFICATION.md",
    "AI_STRATEGY_DISCUSSION.md",
    "ROADMAP.md",
    "IMPROVEMENTS_COMPLETED.md",
    "IMPROVEMENTS.md",
    "MATHEMATICAL_RIGOR_AUDIT.md",
    "AUTONOMOUS_TRADING_AUDIT_PhD_Gaps.md",
    "AUTONOMOUS_TRADING_STATUS_REPORT.md",
    "PhD_AUTONOMOUS_TRADING_ENHANCEMENTS.md",
    "PhD_ENHANCEMENTS_IMPLEMENTED.md",
    "BEFORE_vs_AFTER_PhD_Upgrade.md",
    "WATCHDOG_ADVISORY_HYBRID_ANALYSIS.md",
    "PHD_ENHANCEMENTS_SUMMARY.md",
    "PHD_LEVEL_ENHANCEMENT_COMPLETION_REPORT.md",
    "QUICK_REFERENCE_PhD_Enhancements.md",
    "PhD_ENHANCEMENTS_FINAL_CHECKLIST.md",
    "RESPONSE_TO_AI_CRITIC.md",
    "SETUP_MACOS.md",
    "JULABA_ML_ACCELERATION_PLAN.md",
    "JULABA_ML_IMPLEMENTATION_SPEC.md",
]

workspace = Path("/home/opc/julaba")

# Combine all markdown files
combined_content = "# JULABA COMPREHENSIVE SYSTEM DOCUMENTATION\n\n"
combined_content += "**Generated**: January 14, 2026\n"
combined_content += "**Status**: Complete System Analysis & Implementation\n\n"
combined_content += "---\n\n## TABLE OF CONTENTS\n\n"

# First pass: collect all files that exist and create TOC
existing_files = []
for md_file in md_files:
    filepath = workspace / md_file
    if filepath.exists():
        existing_files.append(md_file)
        combined_content += f"- {md_file}\n"

combined_content += "\n---\n\n"

# Second pass: combine content
for md_file in existing_files:
    filepath = workspace / md_file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add file separator
        combined_content += f"\n\n{'='*80}\n"
        combined_content += f"## {md_file}\n"
        combined_content += f"{'='*80}\n\n"
        combined_content += content
        
        print(f"✅ Added: {md_file}")
    except Exception as e:
        print(f"❌ Error reading {md_file}: {e}")

# Save combined markdown
combined_md_path = workspace / "JULABA_COMPLETE_DOCUMENTATION.md"
with open(combined_md_path, 'w', encoding='utf-8') as f:
    f.write(combined_content)

print(f"\n✅ Combined markdown saved: {combined_md_path}")
print(f"   Size: {len(combined_content):,} bytes")

# Convert to PDF using pandoc
pdf_path = workspace / "JULABA_COMPLETE_DOCUMENTATION.pdf"

print(f"\n📄 Converting to PDF...")
try:
    # Check if pandoc is installed
    result = subprocess.run(
        ["pandoc", "--version"],
        capture_output=True,
        timeout=5
    )
    
    if result.returncode == 0:
        # Use pandoc to convert
        subprocess.run(
            [
                "pandoc",
                str(combined_md_path),
                "-o", str(pdf_path),
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
                "-V", "fontsize=11pt"
            ],
            check=True,
            timeout=60
        )
        print(f"✅ PDF created: {pdf_path}")
    else:
        raise Exception("Pandoc not available, trying alternative...")
        
except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
    print(f"⚠️ Pandoc conversion failed: {e}")
    print(f"   Trying alternative method with wkhtmltopdf...")
    
    try:
        # Try with wkhtmltopdf as fallback
        subprocess.run(
            ["wkhtmltopdf", str(combined_md_path), str(pdf_path)],
            check=True,
            timeout=60
        )
        print(f"✅ PDF created: {pdf_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"⚠️ wkhtmltopdf also not available")
        print(f"   Markdown combined successfully, PDF conversion requires:")
        print(f"   - pandoc: sudo apt-get install pandoc")
        print(f"   - OR wkhtmltopdf: sudo apt-get install wkhtmltopdf")

# Delete original markdown files (keep combined one)
print(f"\n🗑️  Deleting original markdown files...")
deleted_count = 0
for md_file in existing_files:
    filepath = workspace / md_file
    try:
        filepath.unlink()
        print(f"   ✅ Deleted: {md_file}")
        deleted_count += 1
    except Exception as e:
        print(f"   ❌ Error deleting {md_file}: {e}")

print(f"\n{'='*80}")
print(f"✅ COMPLETE!")
print(f"{'='*80}")
print(f"Files combined: {len(existing_files)}")
print(f"Files deleted: {deleted_count}")
print(f"\n📄 Combined markdown: {combined_md_path}")
if pdf_path.exists():
    pdf_size = pdf_path.stat().st_size / 1024 / 1024
    print(f"📄 PDF file: {pdf_path} ({pdf_size:.1f} MB)")
else:
    print(f"📄 PDF file: Not created (requires pandoc/wkhtmltopdf)")

print(f"\n✅ All original .md files have been deleted")
print(f"✅ Keep only: JULABA_COMPLETE_DOCUMENTATION.md (and .pdf if created)")
