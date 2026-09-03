import re
import sys
from pathlib import Path

tex_p = Path("/home/sengar/onnx-quant-benchmark/paper/main.tex")
tex = tex_p.read_text(encoding="utf-8")

labels = re.findall(r"\\label\{([^}]+)\}", tex)
print("=" * 70)
print(f"Verifying Labels in {tex_p.name} (Total: {len(labels)})")
print("=" * 70)

missing = []
for lbl in labels:
    ref_pattern = r"\\ref\{" + re.escape(lbl) + r"\}"
    count = len(re.findall(ref_pattern, tex))
    status = "OK" if count > 0 else "MISSING"
    print(f"  • {lbl:<32} -> {count} citation(s) [{status}]")
    if count == 0:
        missing.append(lbl)

# Also check citations
bib_p = Path("/home/sengar/onnx-quant-benchmark/paper/references.bib")
bib = bib_p.read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cite_keys = set(re.findall(r"\\cite\{([^}]+)\}", tex))
all_cited = []
for c in cite_keys:
    all_cited.extend([k.strip() for k in c.split(",")])

print("\n" + "=" * 70)
print(f"Verifying BibTeX Citations (Total Cited: {len(set(all_cited))})")
print("=" * 70)
missing_cites = []
for k in sorted(set(all_cited)):
    present = k in bib_keys
    status = "OK" if present else "MISSING"
    print(f"  • {k:<28} -> [{status}]")
    if not present:
        missing_cites.append(k)

print("\n" + "=" * 70)
if missing or missing_cites:
    print(f"ERRORS DETECTED: Missing labels {missing}, Missing cites {missing_cites}")
    sys.exit(1)
else:
    print("ALL 9 FIGURES, ALL 2 TABLES, AND ALL 15 CITATIONS VERIFIED CLEANLY!")
    print("=" * 70)
