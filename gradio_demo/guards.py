"""Rule-based input checks that run BEFORE the model.

The fine-tuned model scores surface fluency and does not verify that an essay answers the question
(it gave ~6.0 to random-word gibberish in my robustness probe). These checks are a guardrail, not a
fix: thresholds were chosen from 1,009 real essays (held-out HF + Kaggle) so that none is refused,
while gibberish and repeated text fall far outside the real range.

check(prompt, essay) -> {"ok": bool, "refusals": [...], "warnings": [...], "stats": {...}}
"""
import re

STOP = set(
    "a an the of and or but to in on at for with by from as is are was were be been being this that these "
    "those it its they them their he she his her we our you your i my me not no do does did have has had "
    "will would can could should may might must than then so such also very more most many much some any "
    "other each both either neither however therefore thus while although though because since if when "
    "which who whom whose what where why how there here about into over under between among through "
    "during before after again further once only own same too just".split()
)

MIN_WORDS_REFUSE, MAX_WORDS_REFUSE = 50, 800   # outside this: refuse
TRAIN_WORDS = (100, 600)                      # outside this: warn (range seen in training)
MIN_TTR = 0.25      # real essays: min 0.347 (n=1,009); gibberish 0.07, repeated sentence 0.04
MAX_4GRAM_REPEAT = 0.40   # real essays: max 0.214; repeated text 0.96
WARN_OVERLAP = 0.20       # real essays: ~p1 = 0.15-0.2, so this is a warning only


def _toks(s):
    return re.findall(r"[a-z']+", s.lower())


def _stem(w):
    return re.sub(r"(ies|es|s|ing|ed)$", "", w) if len(w) > 4 else w


def _content(s):
    return {_stem(w) for w in _toks(s) if w not in STOP and len(w) > 2}


def stats(prompt, essay):
    w = _toks(essay)
    n = len(w)
    four = [tuple(w[i:i + 4]) for i in range(max(0, n - 3))]
    pc = _content(prompt)
    return {
        "words": n,
        "type_token_ratio": len(set(w)) / max(n, 1),
        "repeat_4gram_share": 1 - len(set(four)) / max(len(four), 1),
        "prompt_overlap": len(pc & _content(essay)) / max(len(pc), 1),
    }


def check(prompt, essay):
    s = stats(prompt, essay)
    refusals, warnings = [], []
    if s["words"] < MIN_WORDS_REFUSE:
        refusals.append(f"Essay is too short to score ({s['words']} words; need at least {MIN_WORDS_REFUSE}).")
    if s["words"] > MAX_WORDS_REFUSE:
        refusals.append(f"Essay is too long ({s['words']} words; the model was trained on 100-600 words).")
    if s["words"] >= MIN_WORDS_REFUSE:
        if s["type_token_ratio"] < MIN_TTR:
            refusals.append("Text looks like gibberish or heavy word repetition (very low vocabulary diversity); "
                            "the model would otherwise give it a mid-range band.")
        if s["repeat_4gram_share"] > MAX_4GRAM_REPEAT:
            refusals.append("Large parts of the text are repeated; not scoring it.")
    if not refusals:
        if not TRAIN_WORDS[0] <= s["words"] <= TRAIN_WORDS[1]:
            warnings.append(f"{s['words']} words is outside the 100-600 range the model saw in training; "
                            "treat the score with extra caution.")
        if s["prompt_overlap"] < WARN_OVERLAP:
            warnings.append(f"Very little wording overlaps with the question ({s['prompt_overlap']:.0%}). "
                            "This model does NOT check whether the essay answers the question, so an off-topic "
                            "essay can still get a mid-range band.")
    return {"ok": not refusals, "refusals": refusals, "warnings": warnings, "stats": s}
