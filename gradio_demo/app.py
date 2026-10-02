"""Gradio demo: IELTS Task 2 overall-band predictor (LoRA-tuned Qwen2.5-0.5B).

Run locally: pip install -r gradio_demo/requirements.txt && python gradio_demo/app.py
The model is loaded from the Hugging Face Hub (MODEL_ID); set MODEL_ID to a local folder to use local weights.
(Hosting this on a Hugging Face Space needs a PRO subscription, so the hosted page is a static project page.)
"""
import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo layout: grader.py lives one level up
from grader import grade, load  # noqa: E402
from guards import check  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Viktor507/ielts-band-predictor-0.5b")
GITHUB = "https://github.com/ahmedvictor507/ielts-essay-grader-lora"
EX = Path(__file__).resolve().parent.parent / "examples"

model, tok = load(MODEL_ID, device="cpu")

DISCLAIMER = f"""
**Read this first.** This predicts an **overall band only** and was trained on **model-generated labels**, not examiner scores.
It is a *moderate ranking signal* (Pearson r ≈ 0.5 on held-out and external essays), **not** a calibrated or official IELTS score.
It rates surface fluency and **does not check that the essay answers the question**; inputs that are clearly gibberish or
repeated are refused by rule-based checks, and off-topic text is only *warned about*.
Details, results and failure modes: [GitHub]({GITHUB}).

<sub>Runs on CPU: expect roughly 10–30 s per essay.</sub>
"""


def run(prompt, essay):
    prompt, essay = (prompt or "").strip(), (essay or "").strip()
    if not prompt or not essay:
        return "### Paste a question and an essay", "", ""
    g = check(prompt, essay)
    s = g["stats"]
    stats_line = (f"words {s['words']} · vocabulary diversity {s['type_token_ratio']:.2f} · "
                  f"repeated 4-grams {s['repeat_4gram_share']:.0%} · wording overlap with question {s['prompt_overlap']:.0%}")
    if not g["ok"]:
        return "### Not scored", "", "\n".join(f"- ⛔ {r}" for r in g["refusals"]) + f"\n\n<sub>{stats_line}</sub>"
    r = grade(model, tok, prompt, essay)
    notes = [f"- ⚠️ {w}" for w in g["warnings"]]
    if r["band"] is None:
        return "### Could not parse a band", r["raw"], "\n".join(notes)
    return (f"### Predicted overall band: **{r['band']}**", r["raw"],
            "\n".join(notes + [f"<sub>{stats_line} · {r['seconds']:.1f}s on CPU</sub>"]))


def ex(name):
    return (EX / f"essay_{name}.txt").read_text().strip()


PROMPT = (EX / "prompt.txt").read_text().strip()
GIBBERISH = " ".join(["the university skill knowledge job student because important people many also should"] * 20)

with gr.Blocks(title="IELTS Band Predictor") as demo:
    gr.Markdown("# ✍️ IELTS Task 2 Band Predictor\nLoRA-tuned Qwen2.5-0.5B · returns the overall band as strict JSON")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Task 2 question", lines=3, value=PROMPT)
            essay = gr.Textbox(label="Essay (100–600 words works best)", lines=14, placeholder="Paste an essay here…")
            btn = gr.Button("Predict band", variant="primary")
        with gr.Column(scale=2):
            band = gr.Markdown("### Predicted overall band: –")
            raw = gr.Textbox(label="Model output (raw JSON)", interactive=False)
            notes = gr.Markdown()
    gr.Examples(
        examples=[[PROMPT, ex("weak")], [PROMPT, ex("mid")], [PROMPT, ex("strong")], [PROMPT, GIBBERISH]],
        inputs=[prompt, essay],
        label="Examples (synthetic essays; the last one is gibberish to show the guardrail)",
    )
    btn.click(run, [prompt, essay], [band, raw, notes], api_name="predict")

if __name__ == "__main__":
    demo.launch()
