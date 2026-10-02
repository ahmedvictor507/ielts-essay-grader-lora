# Local Gradio demo

Same model, with rule-based input checks (`guards.py`) in front of it: gibberish / heavy repetition / too-short or too-long
essays are refused, off-topic essays get a warning. The checks are a guardrail, not a fix: the thresholds were chosen from
1,009 real essays (0 refused, 2.2% warned), and the model itself still does not verify that an essay answers the question.

```bash
pip install -r gradio_demo/requirements.txt     # install torch for your hardware first if needed
python gradio_demo/app.py                       # downloads Viktor507/ielts-band-predictor-0.5b from the Hub
```
