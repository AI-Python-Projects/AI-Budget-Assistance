"""
report.py
Generates a single HTML report containing every chart produced by
train_model.py, test_model.py, make_budget.py and budget_assistant.py.
Open outputs/report.html in your browser after running it.
"""
import os
import webbrowser
from datetime import datetime
from pathlib import Path

from config import OUTPUT_DIR


def img_block(filename, caption=None):
    """Return an <img> tag if the file exists, else a friendly placeholder."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return (f'<div class="missing">'
                f'[missing] {filename} — run the script that generates it first.'
                f'</div>')
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<img src="{filename}" alt="{filename}">{cap}'


now = datetime.now().strftime("%Y-%m-%d %H:%M")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Budget Assistant — Full Report</title>
<style>
  body {{
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    max-width: 1150px;
    margin: 30px auto;
    padding: 0 24px 80px;
    color: #2c3e50;
    line-height: 1.55;
  }}
  h1 {{
    border-bottom: 3px solid #2c3e50;
    padding-bottom: 10px;
  }}
  h2 {{
    color: #34495e;
    border-bottom: 1px solid #eee;
    padding-bottom: 6px;
    margin-top: 48px;
  }}
  h3 {{
    color: #7f8c8d;
    margin-top: 28px;
  }}
  img {{
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 8px;
    margin: 12px 0;
    background: #fff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .caption {{
    color: #7f8c8d;
    font-size: 0.9em;
    margin-top: -4px;
  }}
  .missing {{
    background: #fff4e5;
    border-left: 4px solid #f39c12;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 12px 0;
    font-family: monospace;
    font-size: 0.9em;
    color: #7f6000;
  }}
  .meta {{
    color: #7f8c8d;
    font-size: 0.9em;
    margin-top: -8px;
  }}
  .toc {{
    background: #f8f9fa;
    border: 1px solid #eee;
    border-radius: 8px;
    padding: 16px 24px;
    margin: 24px 0 0;
  }}
  .toc ul {{
    margin: 8px 0 0;
    columns: 2;
    -webkit-columns: 2;
    padding-left: 20px;
  }}
  .toc a {{ color: #2980b9; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<h1>💰 AI Budget Assistant — Full Report</h1>
<p class="meta">Generated: {now}</p>

<div class="toc">
  <strong>Contents</strong>
  <ul>
    <li><a href="#sec-train">1. Training</a></li>
    <li><a href="#sec-test">2. Testing</a></li>
    <li><a href="#sec-budget">3. Auto Budget Generation</a></li>
    <li><a href="#sec-assist">4. Budget Assistant Output</a></li>
  </ul>
</div>