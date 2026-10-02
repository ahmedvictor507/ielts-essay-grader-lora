"""Grade the three bundled demo essays and print a compact report (used to record the README GIF)."""
import sys
from pathlib import Path

from grader import grade, load

model_dir = sys.argv[1] if len(sys.argv) > 1 else "model-0.5B"
ex = Path("examples")
prompt = (ex / "prompt.txt").read_text().strip()

print("loading model ...", flush=True)
model, tok = load(model_dir)
grade(model, tok, prompt, (ex / "essay_mid.txt").read_text())  # warm-up (not shown)
print(f"\nPROMPT: {prompt}\n")
for name in ("weak", "mid", "strong"):
    essay = (ex / f"essay_{name}.txt").read_text()
    r = grade(model, tok, prompt, essay)
    first = essay.strip().split(".")[0][:70]
    print(f'essay_{name:<6} "{first}..."')
    print(f"  -> {r['raw']}   ({r['words']} words, {r['seconds']:.2f}s)")
