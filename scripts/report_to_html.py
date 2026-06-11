"""Render results/report/REPORT.md to a self-contained, printable HTML file.

Embeds all referenced figures as base64 (so the file is portable) and adds
print CSS with page breaks before each top-level section. Open the HTML in a
browser and use Print -> Save as PDF for a paginated ~10-page document.

Handles the markdown subset used in REPORT.md: ATX headings, --- rules,
> blockquotes, ![img](path), GitHub pipe tables, - and n. lists, and inline
**bold** / *italic* / `code`. No external markdown dependency.

Run:  python scripts/report_to_html.py
"""
import os
import re
import sys
import base64
import html as htmllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(REPO, "results", "report")
SRC = os.path.join(REPORT_DIR, "REPORT.md")
DST = os.path.join(REPORT_DIR, "REPORT.html")


def inline(text):
    text = htmllib.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def embed_image(relpath, alt):
    path = os.path.normpath(os.path.join(REPORT_DIR, relpath))
    if not os.path.isfile(path):
        return f'<p class="missing">[missing figure: {htmllib.escape(relpath)}]</p>'
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{htmllib.escape(alt)}"/>'
            f'</figure>')


def render(md):
    lines = md.split("\n")
    out = []
    i = 0
    n = len(lines)

    def close_list(stack):
        while stack:
            out.append(f"</{stack.pop()}>")

    list_stack = []

    while i < n:
        line = lines[i]

        # blank
        if not line.strip():
            close_list(list_stack)
            i += 1
            continue

        # horizontal rule -> section page break handled by h2 CSS; render hr
        if re.fullmatch(r"-{3,}", line.strip()):
            close_list(list_stack)
            out.append('<hr/>')
            i += 1
            continue

        # image  ![alt](path)
        m = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if m:
            close_list(list_stack)
            out.append(embed_image(m.group(2), m.group(1)))
            i += 1
            continue

        # heading
        m = re.match(r"(#{1,6})\s+(.*)", line)
        if m:
            close_list(list_stack)
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        # blockquote (possibly multi-line)
        if line.lstrip().startswith(">"):
            close_list(list_stack)
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(f"<blockquote>{inline(' '.join(buf))}</blockquote>")
            continue

        # table: a line with | and a following |---| separator
        if "|" in line and i + 1 < n and re.search(r"\|?\s*:?-{3,}", lines[i + 1]) and "|" in lines[i + 1]:
            close_list(list_stack)
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2  # skip header + separator
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            thead = "".join(f"<th>{inline(c)}</th>" for c in header)
            body = ""
            for r in rows:
                body += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
            out.append(f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>")
            continue

        # unordered list
        m = re.match(r"(\s*)-\s+(.*)", line)
        if m:
            if not list_stack or list_stack[-1] != "ul":
                close_list(list_stack)
                out.append("<ul>"); list_stack.append("ul")
            out.append(f"<li>{inline(m.group(2))}</li>")
            i += 1
            continue

        # ordered list
        m = re.match(r"(\s*)\d+\.\s+(.*)", line)
        if m:
            if not list_stack or list_stack[-1] != "ol":
                close_list(list_stack)
                out.append("<ol>"); list_stack.append("ol")
            out.append(f"<li>{inline(m.group(2))}</li>")
            i += 1
            continue

        # paragraph (gather until blank)
        close_list(list_stack)
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"(#{1,6}\s|!\[|>|\s*-\s|\s*\d+\.\s|-{3,}$)", lines[i]) and "|" not in lines[i]:
            buf.append(lines[i]); i += 1
        out.append(f"<p>{inline(' '.join(buf))}</p>")

    close_list(list_stack)
    return "\n".join(out)


CSS = """
:root { --ink:#1a1a1a; --muted:#666; --rule:#ddd; --accent:#b5450f; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; color: var(--ink);
       max-width: 820px; margin: 0 auto; padding: 32px 40px; line-height: 1.5;
       font-size: 14px; }
h1 { font-size: 26px; border-bottom: 3px solid var(--accent); padding-bottom: 8px; }
h2 { font-size: 19px; margin-top: 1.6em; border-bottom: 1px solid var(--rule);
     padding-bottom: 4px; page-break-before: always; }
h1 + h2, h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 15.5px; color: var(--accent); margin-top: 1.3em; }
h4 { font-size: 14px; color: var(--muted); }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12.5px;
        page-break-inside: avoid; }
th, td { border: 1px solid var(--rule); padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f4f4f4; }
tbody tr:nth-child(even) { background: #fafafa; }
figure { margin: 14px 0; text-align: center; page-break-inside: avoid; }
img { max-width: 100%; border: 1px solid #eee; }
blockquote { border-left: 4px solid var(--accent); margin: 12px 0; padding: 6px 14px;
             background: #fbf6f2; color: #333; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 92%;
       font-family: 'Cascadia Code', Consolas, monospace; }
hr { border: none; border-top: 1px solid var(--rule); margin: 18px 0; }
ul, ol { margin: 8px 0 8px 4px; padding-left: 22px; }
li { margin: 3px 0; }
.missing { color: #c00; font-style: italic; }
@media print { body { padding: 0 12px; } a { color: inherit; text-decoration: none; } }
@page { size: A4; margin: 18mm 16mm; }
"""


def main():
    with open(SRC, "r", encoding="utf-8") as fh:
        md = fh.read()
    body = render(md)
    doc = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
           f"<title>Reward Encoding in the Human Brain — Report</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")
    with open(DST, "w", encoding="utf-8") as fh:
        fh.write(doc)
    size_kb = os.path.getsize(DST) / 1024
    print(f"Saved -> {DST}  ({size_kb:.0f} KB, figures embedded)")
    print("Open in a browser and Print -> Save as PDF for a paginated document.")


if __name__ == "__main__":
    main()
