import re
import fitz

def check_log():
    with open('manuscript/paper.log', 'r', encoding='utf-8', errors='ignore') as f:
        log = f.read()

    undef_cites = re.findall(r"Citation [^\n]+ undefined", log)
    undef_refs = re.findall(r"Reference [^\n]+ undefined", log)
    overfull = [line for line in log.split('\n') if 'Overfull \\hbox' in line]
    underfull = [line for line in log.split('\n') if 'Underfull \\hbox' in line]
    errors = [line for line in log.split('\n') if line.startswith('!')]

    print("=== LATEX COMPILATION LOG AUDIT ===")
    print(f"Errors (!): {len(errors)}")
    for e in errors[:5]:
        print(f"  {e}")
    print(f"Undefined citations: {len(undef_cites)}")
    for c in undef_cites:
        print(f"  {c}")
    print(f"Undefined references: {len(undef_refs)}")
    for r in undef_refs:
        print(f"  {r}")
    print(f"Overfull \\hbox: {len(overfull)}")
    for o in overfull:
        print(f"  {o}")
    print(f"Underfull \\hbox: {len(underfull)}")

def check_structure():
    with open('manuscript/paper.tex', 'r', encoding='utf-8') as f:
        tex = f.read()

    print("\n=== CITATIONS AUDIT ===")
    cites = re.findall(r"\\cite\{([^}]+)\}", tex)
    all_keys = []
    for c in cites:
        for k in c.split(','):
            k = k.strip()
            if k: all_keys.append(k)
    unique_keys = list(dict.fromkeys(all_keys))

    with open('manuscript/references.bib', 'r', encoding='utf-8') as f:
        bib = f.read()
    bib_keys = re.findall(r"@\w+\{([^,]+),", bib)

    print(f"Cited keys in paper: {len(unique_keys)}")
    print(f"Bib keys in references.bib: {len(bib_keys)}")
    missing = [k for k in unique_keys if k not in bib_keys]
    unused = [k for k in bib_keys if k not in unique_keys]
    print(f"Missing from bib: {missing}")
    print(f"Unused in bib: {unused}")

    print("\n=== FIGURE CALLOUTS AUDIT ===")
    figs_defined = re.findall(r"\\label\{(fig:[^}]+)\}", tex)
    figs_called = re.findall(r"\\ref\{(fig:[^}]+)\}", tex)
    print(f"Figures defined: {figs_defined}")
    print(f"Figures called: {set(figs_called)}")
    for f in figs_defined:
        if f not in figs_called:
            print(f"WARNING: {f} defined but never referenced via \\ref!")

    print("\n=== TABLE CALLOUTS AUDIT ===")
    tabs_defined = re.findall(r"\\label\{(tab:[^}]+)\}", tex)
    tabs_called = re.findall(r"\\ref\{(tab:[^}]+)\}", tex)
    print(f"Tables defined: {tabs_defined}")
    print(f"Tables called: {set(tabs_called)}")
    for t in tabs_defined:
        if t not in tabs_called:
            print(f"WARNING: {t} defined but never referenced via \\ref!")

    print("\n=== EQUATION CALLOUTS AUDIT ===")
    eqs_defined = re.findall(r"\\label\{(eq:[^}]+)\}", tex)
    eqs_called = re.findall(r"\\ref\{(eq:[^}]+)\}", tex)
    print(f"Equations defined: {eqs_defined}")
    print(f"Equations called: {set(eqs_called)}")
    for e in eqs_defined:
        if e not in eqs_called:
            print(f"NOTICE: {e} defined but not explicitly called via \\ref")

    print("\n=== SPELLING & TYPO CHECKS ===")
    typo_candidates = [
        'thre', 'refernces', 'wighted', 'lebel', 'croped', 'plagarism',
        'detction', 'boldword', 'humanize', 'inter-patinet', 'wavelete',
        'atention', 'artifcat', 'arrhytmia', 'electrocardiagram', 'resutls',
        'perfromance', 'sensitiivty', 'specifity', 'databse'
    ]
    for w in typo_candidates:
        if re.search(r'\b' + re.escape(w) + r'\b', tex, re.IGNORECASE):
            print(f"POTENTIAL TYPO FOUND: {w}")

if __name__ == '__main__':
    check_log()
    check_structure()
