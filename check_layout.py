import fitz
import subprocess
import os

def compile_and_check():
    cmd = 'powershell -Command "cd manuscript; pdflatex -interaction=nonstopmode paper.tex; bibtex paper; pdflatex -interaction=nonstopmode paper.tex; pdflatex -interaction=nonstopmode paper.tex; cd ..; Copy-Item manuscript\\paper.pdf -Destination WaveCrossNet_Publication.pdf -Force"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    doc = fitz.open('WaveCrossNet_Publication.pdf')
    print(f"Total pages: {len(doc)}")
    for i, p in enumerate(doc):
        blocks = p.get_text('blocks')
        words = len(p.get_text().split())
        chars = len(p.get_text())
        last_y = max([b[3] for b in blocks]) if blocks else 0
        print(f"Page {i+1}: {words} words, {chars} chars, last block y={last_y:.1f}")
    if len(doc) >= 6:
        print("\n--- Page 6 Blocks ---")
        p6_blocks = doc[5].get_text('blocks')
        for b in p6_blocks:
            clean_text = b[4].strip().replace('\n', ' ')[:45]
            clean_text = clean_text.encode('ascii', 'replace').decode('ascii')
            print(f"  Col {'1' if b[0] < 300 else '2'} ({b[0]:.1f}, {b[1]:.1f} -> {b[2]:.1f}, {b[3]:.1f}): {clean_text}")

if __name__ == '__main__':
    compile_and_check()
