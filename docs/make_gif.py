"""Render docs/demo.gif from the REAL captured output of `python demo.py` (no invented text).

Usage: python docs/make_gif.py docs/demo_output.txt docs/demo.gif
Playback is sped up (loading/inference pauses are shortened); the text and timings shown are real.
"""
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1], sys.argv[2]
captured = open(src).read().rstrip("\n").split("\n")

W, H, PAD = 980, 470, 18
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)
CW = FONT.getbbox("M")[2]
COLS = (W - 2 * PAD) // CW
LH = 21
BG, FG, DIM, GREEN, CYAN, YEL = "#14161b", "#d7dae0", "#7f848e", "#98c379", "#56b6c2", "#e5c07b"

cmd = "$ python demo.py"
lines = []  # (text, colour, bold)
for raw in captured:
    colour = FG
    if raw.startswith("loading"):
        colour = DIM
    elif raw.startswith("PROMPT"):
        colour = CYAN
    elif raw.strip().startswith("->"):
        colour = GREEN
    elif raw.startswith("essay_"):
        colour = YEL
    wrapped = textwrap.wrap(raw, COLS, subsequent_indent="        ", drop_whitespace=False) or [""]
    lines += [(w, colour, colour == GREEN) for w in wrapped]


def frame(typed, shown, cursor=True):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 34], fill="#21252b")
    for i, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        d.ellipse([14 + i * 22, 11, 26 + i * 22, 23], fill=c)
    d.text((W // 2 - 230, 8), "Jetson Orin Nano  |  Qwen2.5-0.5B + LoRA (fp16)  |  real run", font=FONT, fill=DIM)
    y = 34 + PAD
    d.text((PAD, y), typed + ("_" if cursor and shown == 0 else ""), font=BOLD, fill=FG)
    y += LH
    for text, colour, bold in lines[:shown]:
        d.text((PAD, y), text, font=BOLD if bold else FONT, fill=colour)
        y += LH
    return im


frames, durs = [], []
for i in range(0, len(cmd) + 1, 2):  # type the command
    frames.append(frame(cmd[:i], 0)); durs.append(70)
frames.append(frame(cmd, 0)); durs.append(400)
# reveal output: loading line, pause, prompt, then each essay header + result with a pause on results
shown = 0
for idx, (text, colour, _) in enumerate(lines):
    shown = idx + 1
    frames.append(frame(cmd, shown, cursor=False))
    if text.startswith("loading"):
        durs.append(1100)
    elif colour == GREEN:
        durs.append(1500)
    elif colour == YEL:
        durs.append(500)
    else:
        durs.append(250)
durs[-1] = 4000

frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
print(f"wrote {dst}: {len(frames)} frames, {sum(durs)/1000:.1f}s")
