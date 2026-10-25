"""
Wello - an AI assistant for academic writing in English.
Built with Streamlit. Runs for free on Streamlit Community Cloud.

Tools
 Grammar     AI + LanguageTool, shown as tracked changes
 Style       rules + AI
 Rubric      AI, scored, with draft-to-draft progress
 Paragraph   AI, topic sentence / support / explanation / link
 Paraphrase  AI
 Plan        AI, thesis options and outline
 APA 7       no AI, formats references and in-text citations
 Insights    no AI, statistics and sentence rhythm
 Phrases     no AI, academic phrasebank
 Feedback    research form
"""

import base64
import csv
import difflib
import html
import json
import os
import random
import re
import time
from datetime import datetime, timezone

import requests
import streamlit as st

LOGO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
HAS_LOGO = os.path.exists(LOGO_FILE)

st.set_page_config(
    page_title="Wello",
    page_icon=LOGO_FILE if HAS_LOGO else "✏️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------
# Configuration (set these in Streamlit "Settings > Secrets")
# --------------------------------------------------------------------------
def get_secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:  # noqa: BLE001
        return os.environ.get(name, default)


def clean_secret(value):
    return str(value).strip().strip("\"'“”‘’").strip()


MODEL_ID = clean_secret(get_secret("MODEL_ID", "openai/gpt-oss-120b"))
FALLBACK_MODEL = clean_secret(get_secret("FALLBACK_MODEL", "openai/gpt-oss-20b"))
GROQ_API_KEY = clean_secret(get_secret("GROQ_API_KEY", ""))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_TOKEN = clean_secret(get_secret("HF_TOKEN", ""))  # optional: keep data in a private HF dataset
FEEDBACK_DATASET = clean_secret(get_secret("FEEDBACK_DATASET", ""))
ADMIN_PASSWORD = str(get_secret("ADMIN_PASSWORD", ""))
LANGUAGETOOL_URL = "https://api.languagetool.org/v2/check"
MAX_WORDS = 1200
COOLDOWN_SECONDS = 3

FEEDBACK_FILE = "feedback.csv"
FEEDBACK_FIELDS = [
    "timestamp", "name", "role", "feature_used",
    "ease_of_use", "usefulness", "accuracy", "design", "comment",
]
USAGE_FILE = "usage_log.csv"
USAGE_FIELDS = ["timestamp", "feature", "words", "status"]

INK = "#1E2A3A"
ACCENT = "#C8650F"


# --------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700&family=Literata:opsz,wght@7..72,400;7..72,500;7..72,600&display=swap');

:root { --ink:#1E2A3A; --muted:#5B6878; --paper:#FFFFFF; --page:#F5F6F8; --edge:#D5DAE1;
        --rule:#D3E2F0; --margin:#E9A3A3; --accent:#C8650F; --add:#1B6E3C; --del:#B3261E; }
html, body, .stApp { background: var(--page); color: var(--ink); font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif; }
.block-container { max-width: 820px; padding-top: 2.6rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }

/* masthead */
.masthead { display:flex; align-items:center; gap:.9rem; margin-bottom:1.1rem; }
.masthead img { width:46px; height:46px; border-radius:11px; display:block; }
.masthead .name { font-size:1.85rem; font-weight:700; letter-spacing:-.02em; line-height:1.05; }
.masthead .line { color: var(--muted); font-size:.98rem; margin-top:.15rem; }

/* tool heading */
.tool-title { font-size:1.55rem; font-weight:700; letter-spacing:-.015em; margin:1.6rem 0 .2rem 0; }
.tool-desc { color: var(--muted); font-size:1rem; line-height:1.5; max-width:62ch; margin:0 0 1.1rem 0; }

/* the writing surface: a ruled exercise-book page with a red margin line */
[class*="st-key-paper"] div[data-baseweb="textarea"],
[class*="st-key-paper"] div[data-baseweb="base-input"] { background: var(--paper); border-radius: 3px; }
[class*="st-key-paper"] textarea {
    font-family: 'Literata', Georgia, 'Times New Roman', serif; font-size: 17px; line-height: 32px;
    padding: 8px 20px 8px 66px; color: var(--ink); background-color: var(--paper);
    background-image:
        linear-gradient(to right, transparent 0, transparent 48px, var(--margin) 48px, var(--margin) 50px, transparent 50px),
        repeating-linear-gradient(to bottom, transparent 0, transparent 31px, var(--rule) 31px, var(--rule) 32px);
    background-attachment: local;
}
[class*="st-key-paper"] textarea::placeholder { color:#8D98A6; font-style: italic; }

/* feedback sheet */
[class*="st-key-sheet"] { background: var(--paper); border:1px solid var(--edge); border-radius:3px; padding:1.1rem 1.4rem; }
[class*="st-key-sheet"] p, [class*="st-key-sheet"] li { font-family:'Literata', Georgia, serif; font-size:1.02rem; line-height:1.7; }
[class*="st-key-sheet"] h3 { font-family:'Bricolage Grotesque', sans-serif; font-size:1.08rem; font-weight:700; margin:1.1rem 0 .3rem 0; padding:0; }
[class*="st-key-sheet"] h3:first-child { margin-top:.2rem; }
[class*="st-key-sheet"] table { font-size:.95rem; }

/* tracked changes */
.track { font-family:'Literata', Georgia, serif; font-size:1.06rem; line-height:1.95; background:var(--paper);
         border:1px solid var(--edge); border-radius:3px; padding:1rem 1.3rem; }
.track del { color:var(--del); text-decoration: line-through; text-decoration-thickness:1.5px; }
.track ins { color:var(--add); text-decoration: underline; text-decoration-thickness:1.5px; text-underline-offset:3px; }
.legend { color:var(--muted); font-size:.88rem; margin:.4rem 0 1rem 0; }

/* rubric */
.total { font-size:2.8rem; font-weight:700; letter-spacing:-.03em; line-height:1; }
.total small { font-size:1.15rem; font-weight:500; color:var(--muted); letter-spacing:0; }
.crit { padding:.7rem 0; border-top:1px solid var(--edge); }
.crit:first-of-type { border-top:none; }
.crit .row { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
.crit .cname { font-weight:600; }
.crit .score { font-weight:700; }
.crit .cmt { font-family:'Literata', Georgia, serif; font-size:.98rem; line-height:1.55; color:#2E3B4D; margin-top:.2rem; }
.tally { display:inline-flex; gap:3px; vertical-align:middle; margin-right:.6rem; }
.tally i { display:block; width:22px; height:7px; border-radius:1px; background:#DCE2EA; }
.tally i.on { background: var(--ink); }

/* statistics */
.stats { display:grid; grid-template-columns:repeat(3, 1fr); gap:1.2rem 1.4rem; margin:.4rem 0 1.2rem 0; }
@media (max-width: 520px) { .stats { grid-template-columns:repeat(2, 1fr); } }
.stats b { display:block; font-size:2rem; font-weight:700; letter-spacing:-.02em; line-height:1.05; }
.stats span { color:var(--muted); font-size:.92rem; }

/* controls */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius:5px; font-weight:600; font-family:'Bricolage Grotesque', sans-serif; border:1px solid var(--edge); background:var(--paper); color:var(--ink);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover { background:#9F4F0A; border-color:#9F4F0A; color:#fff; }
.counter { color:var(--muted); font-size:.88rem; padding-top:.45rem; }
.chip { display:inline-block; border:1px solid var(--edge); background:var(--paper); border-radius:3px;
        padding:.12rem .55rem; margin:0 .4rem .4rem 0; font-size:.88rem; }
.chip b { color:var(--accent); margin-left:.15rem; }
.eyebrow { color:var(--muted); font-size:.78rem; letter-spacing:.09em; text-transform:uppercase; font-weight:600;
           margin:1.1rem 0 .4rem 0; }
.note { color:var(--muted); font-size:.9rem; max-width:64ch; }
a { color: var(--accent); }
</style>
""",
    unsafe_allow_html=True,
)


def logo_img():
    if not HAS_LOGO:
        return ""
    with open(LOGO_FILE, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="Wello logo">'


# --------------------------------------------------------------------------
# Data logging (feedback + anonymous usage statistics)
# --------------------------------------------------------------------------
def push_to_dataset(filename, fields, row):
    """Optionally keep a permanent copy in a private HF dataset repo."""
    if not (FEEDBACK_DATASET and HF_TOKEN):
        return
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=HF_TOKEN)
        rows = []
        try:
            path = hf_hub_download(
                repo_id=FEEDBACK_DATASET, filename=filename,
                repo_type="dataset", token=HF_TOKEN, force_download=True,
            )
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:  # noqa: BLE001
            rows = []
        rows.append(row)
        tmp = f"upload_{filename}"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        api.upload_file(path_or_fileobj=tmp, path_in_repo=filename,
                        repo_id=FEEDBACK_DATASET, repo_type="dataset")
    except Exception:  # noqa: BLE001
        pass


def append_csv(filename, fields, row):
    new_file = not os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def log_usage(feature, words, status):
    """Anonymous statistics only: tool name, word count and result. The text itself is never stored."""
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "feature": feature, "words": words, "status": status}
    try:
        append_csv(USAGE_FILE, USAGE_FIELDS, row)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# AI helper (Groq, open-source models) with automatic fallback
# --------------------------------------------------------------------------
class AIError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _post_groq(model, system_prompt, user_prompt, temperature, max_tokens, json_mode=False):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_completion_tokens": max_tokens + 1500,
    }
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"  # faster answers, saves free quota
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)
        if r.status_code == 400 and json_mode:  # model without JSON mode: ask again as plain text
            payload.pop("response_format", None)
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise AIError(None, f"{type(exc).__name__}: {exc}") from exc
    if r.status_code != 200:
        raise AIError(r.status_code, r.text[:400])
    content = (r.json()["choices"][0]["message"].get("content") or "").strip()
    if not content:
        raise AIError(None, "The model returned an empty answer.")
    return content


@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def _cached_post(model, system_prompt, user_prompt, temperature, max_tokens, json_mode):
    return _post_groq(model, system_prompt, user_prompt, temperature, max_tokens, json_mode)


HINTS = {
    400: "The request was rejected. The model name in MODEL_ID may be wrong.",
    401: "The API key is missing or invalid. Check GROQ_API_KEY in Secrets.",
    403: "Access denied. Check the Groq account and API key.",
    404: "The model is not available. Set MODEL_ID in Secrets to another model.",
    413: "The text is too long for the free limit. Use a shorter text.",
    429: "The free usage limit is reached. Wait a minute, then try again.",
}


def llm(system_prompt, user_prompt, temperature=0.3, max_tokens=1800, use_cache=True, json_mode=False):
    """Call the AI. Returns (text, ok). Tries a fallback model if the main one is busy."""
    call = _cached_post if use_cache else _post_groq
    models = [MODEL_ID] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL_ID else [])
    last_error = None
    for model in models:
        try:
            return call(model, system_prompt, user_prompt, temperature, max_tokens, json_mode), True
        except AIError as exc:
            last_error = exc
            if exc.status in (401, 403):  # another model will not help
                break
    detail = last_error.detail.replace(GROQ_API_KEY, "***") if GROQ_API_KEY else last_error.detail
    message = (
        "The AI service did not respond. "
        f"{HINTS.get(last_error.status, 'Wait a moment, then try again.')}\n\n"
        f"Details: status {last_error.status}, key length {len(GROQ_API_KEY)}.\n\n"
        f"```\n{detail[:300]}\n```"
    )
    return message, False


def validate_text(text):
    text = (text or "").strip()
    if not text:
        return None, "Paste or type your text first."
    if len(text.split()) > MAX_WORDS:
        return None, f"The text is over {MAX_WORDS} words. Shorten it and try again."
    return text, None


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


# --------------------------------------------------------------------------
# Grammar
# --------------------------------------------------------------------------
def languagetool_check(text):
    try:
        r = requests.post(LANGUAGETOOL_URL, data={"text": text, "language": "en-US"}, timeout=20)
        r.raise_for_status()
        return r.json().get("matches", [])
    except Exception:  # noqa: BLE001
        return None


CORRECTED_RE = re.compile(r"###\s*Corrected Text\s*\n(.*?)(?=\n###|\Z)", re.S | re.I)
ERROR_TAGS = ["Verb tense", "Subject-verb agreement", "Verb form", "Article", "Preposition", "Word choice",
              "Spelling", "Punctuation", "Sentence structure", "Plural or singular", "Other"]
TAG_LIST = ", ".join("[" + t + "]" for t in ERROR_TAGS)
TAG_RE = re.compile(r"\[(" + "|".join(re.escape(t) for t in ERROR_TAGS) + r")\]")


def check_grammar(text):
    text, error = validate_text(text)
    if error:
        return {"md": error}, False

    matches = languagetool_check(text)
    detected = ""
    if matches:
        lines = []
        for m in matches[:40]:
            start, length = m["offset"], m["length"]
            suggestions = ", ".join(s["value"] for s in m.get("replacements", [])[:3]) or "-"
            lines.append(f"- **{text[start:start + length]}** → {suggestions} ({m.get('message', '')})")
        detected = "\n".join(lines)

    system = (
        "You are a friendly English writing tutor for university students. "
        "Correct grammar, spelling, punctuation and word-choice errors without changing the author's meaning. "
        "Explain every mistake in simple English so the student can learn."
    )
    user = f"""Check the text below.

Possible errors found by an automatic checker (may contain false alarms, verify them):
{detected if detected else "None found or checker unavailable."}

Reply in exactly this format:

### Corrected Text
(the full corrected text, nothing else, no quotation marks)

### Explanation of Corrections
(a numbered list. Start every item with exactly one error tag in square brackets, chosen from this list: {TAG_LIST}. After the tag write: ~~wrong~~ → **correct**, the grammar rule name, and a one-sentence simple explanation)

### Quick Tip
(one short tip to avoid the most frequent mistake)

If the text has no errors, say so and give one suggestion for improvement.

TEXT:
\"\"\"{text}\"\"\""""
    md, ok = llm(system, user, temperature=0.2)
    payload = {"md": md, "original": text}
    if ok:
        m = CORRECTED_RE.search(md)
        if m:
            payload["corrected"] = m.group(1).strip().strip("\"“”")
        tags = {}
        for tag in TAG_RE.findall(section(md, "Explanation of Corrections")):
            tags[tag] = tags.get(tag, 0) + 1
        payload["tags"] = dict(sorted(tags.items(), key=lambda kv: -kv[1]))
    return payload, ok


TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]|\s+")


def tracked_changes_html(original, corrected):
    """Word-level diff: removed words struck through, added words underlined."""
    a, b = TOKEN_RE.findall(original), TOKEN_RE.findall(corrected)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out, changes = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old, new = "".join(a[i1:i2]), "".join(b[j1:j2])
        if tag == "equal":
            out.append(html.escape(old))
            continue
        changes += 1
        if old.strip():
            out.append(f"<del>{html.escape(old.strip())}</del>")
        if new.strip():
            out.append(f" <ins>{html.escape(new.strip())}</ins>")
        if old.endswith(" ") or new.endswith(" "):
            out.append(" ")
    return "".join(out), changes


# --------------------------------------------------------------------------
# Academic style
# --------------------------------------------------------------------------
INFORMAL_PATTERNS = {
    r"\ba lot of\b": "many / numerous / a considerable amount of",
    r"\blots of\b": "many / numerous / substantial",
    r"\bkind of\b": "somewhat / rather",
    r"\bsort of\b": "somewhat / to some extent",
    r"\bstuff\b": "materials / items / (be specific)",
    r"\bthings\b": "factors / aspects / elements (be specific)",
    r"\bthing\b": "factor / aspect / element (be specific)",
    r"\bget\b": "obtain / receive / become",
    r"\bgot\b": "obtained / received / became",
    r"\bbig\b": "significant / substantial / considerable",
    r"\bgood\b": "beneficial / effective / favorable",
    r"\bbad\b": "negative / harmful / unfavorable",
    r"\breally\b": "(delete) or markedly / considerably",
    r"\bvery\b": "(delete) or a stronger adjective, e.g. very important → crucial",
    r"\bpretty\b": "fairly / relatively",
    r"\bnowadays\b": "currently / in recent years",
    r"\band so on\b": "(list all items or use 'for example')",
    r"\betc\.?": "(list all items or use 'such as')",
    r"\bbesides\b": "furthermore / moreover",
    r"\bbut\b": "however / nevertheless (at the start of a sentence)",
    r"\banyways?\b": "nevertheless / nonetheless",
    r"\bokay\b|\bok\b": "acceptable / satisfactory",
    r"\bguys\b": "individuals / people / participants",
    r"\bkids\b": "children",
}
CONTRACTION_PATTERN = r"\b\w+(?:n't|'re|'ve|'ll|'m|'d)\b|\b(?:it's|that's|there's|let's|what's)\b"
PERSONAL_PATTERN = r"\b(?:I|me|my|you|your|we|our)\b"


def rule_based_style_scan(text):
    findings = []
    contractions = sorted(set(m.group(0) for m in re.finditer(CONTRACTION_PATTERN, text, flags=re.I)))
    if contractions:
        findings.append(f"- **Contractions** ({', '.join(contractions[:10])}): write them in full (*do not*, *it is*).")
    for pattern, suggestion in INFORMAL_PATTERNS.items():
        found = sorted(set(m.group(0).lower() for m in re.finditer(pattern, text, flags=re.I)))
        if found:
            findings.append(f"- **{found[0]}**: consider {suggestion}")
    personal = sorted(set(m.group(0) for m in re.finditer(PERSONAL_PATTERN, text)))
    if personal:
        findings.append(
            f"- **Personal pronouns** ({', '.join(personal[:8])}): many academic genres prefer an impersonal style "
            "(*This study argues...* instead of *I think...*). Check your lecturer's guideline."
        )
    return "\n".join(findings)


def check_style(text):
    text, error = validate_text(text)
    if error:
        return {"md": error}, False

    scan = rule_based_style_scan(text)
    system = (
        "You are an academic writing coach. You evaluate whether a text follows formal academic English conventions: "
        "formality, objectivity, precision, hedging, concision, and avoidance of colloquial language."
    )
    user = f"""Evaluate the academic style of the text below.

An automatic scan already found these informal features:
{scan if scan else "None."}

Reply in exactly this format:

### Overall Style Level
(one of: Informal / Semi-formal / Academic. Add one sentence of justification.)

### Problems and Better Alternatives
(a numbered list. For each: the original phrase → a more academic alternative, with a short reason. Include issues the scan missed, such as vague words, overly strong claims that need hedging, wordiness, or repetition.)

### Academically Revised Version
(rewrite the whole text in a formal academic style, keeping the meaning)

TEXT:
\"\"\"{text}\"\"\""""
    ai_part, ok = llm(system, user, temperature=0.3, max_tokens=2200)
    header = "### Found by the scan\n" + (scan if scan else "No obvious informal expressions.") + "\n\n"
    payload = {"md": header + ai_part}
    if ok:
        payload["revised"] = section(ai_part, "Academically Revised Version")
    return payload, ok


# --------------------------------------------------------------------------
# Rubric
# --------------------------------------------------------------------------
CRITERIA = [
    "Content and main idea",
    "Organization and structure",
    "Coherence and cohesion",
    "Vocabulary and academic style",
    "Grammar and mechanics",
]


def rubric_feedback(text, genre, task):
    text, error = validate_text(text)
    if error:
        return {"md": error}, False

    task_line = (f"\nThe writing task was: {task.strip()}\nAlso judge how well the text answers it.\n"
                 if task and task.strip() else "")
    system = (
        "You are an experienced academic writing lecturer. You assess student writing fairly using a rubric, "
        "give specific evidence from the text, and offer practical suggestions. Be encouraging but honest. "
        "You reply with JSON only."
    )
    user = f"""Assess the following {genre}. Score each of the five criteria from 1 (poor) to 5 (excellent).
{task_line}
Criteria, in this order:
1. Content and main idea (clear thesis or purpose, relevant and sufficient support)
2. Organization and structure (introduction, topic sentences, logical order, conclusion)
3. Coherence and cohesion (transitions, flow, reference words)
4. Vocabulary and academic style (precision, formality, variety)
5. Grammar and mechanics (accuracy, punctuation, spelling)

Reply with one JSON object and nothing else, in this shape:
{{"scores": [int, int, int, int, int],
  "comments": ["one or two sentences with evidence from the text, for each criterion, same order"],
  "strengths": ["2 or 3 short points"],
  "improve": ["3 or 4 points, each with a concrete example from the text and how to fix it"],
  "next_steps": ["three prioritized actions for the next revision"]}}

TEXT:
\"\"\"{text}\"\"\""""
    raw, ok = llm(system, user, temperature=0.2, max_tokens=1800, json_mode=True)
    if not ok:
        return {"md": raw}, False
    data = parse_json(raw)
    try:
        scores = [max(1, min(5, int(round(float(s))))) for s in data["scores"]][:5]
        comments = [str(c) for c in data["comments"]][:5]
        if len(scores) != 5:
            raise ValueError("need five scores")
        comments += [""] * (5 - len(comments))
        rubric = {
            "scores": scores, "comments": comments,
            "strengths": [str(x) for x in data.get("strengths", [])][:4],
            "improve": [str(x) for x in data.get("improve", [])][:5],
            "next_steps": [str(x) for x in data.get("next_steps", [])][:4],
            "words": len(text.split()),
        }
        return {"md": "", "rubric": rubric}, True
    except Exception:  # noqa: BLE001
        return {"md": raw}, True  # show the model's answer as it came


def rubric_level(total):
    if total >= 22:
        return "Advanced"
    if total >= 17:
        return "Proficient"
    if total >= 11:
        return "Developing"
    return "Beginner"


def rubric_as_text(r):
    total = sum(r["scores"])
    lines = [f"Wello rubric feedback: {total} / 25 ({rubric_level(total)})", ""]
    for name, score, cmt in zip(CRITERIA, r["scores"], r["comments"]):
        lines.append(f"{name}: {score} / 5")
        lines.append(f"  {cmt}")
    for title, key in (("Strengths", "strengths"), ("To improve", "improve"), ("Next steps", "next_steps")):
        lines += ["", title + ":"] + [f"- {x}" for x in r[key]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Paragraph check
# --------------------------------------------------------------------------
def check_paragraph(text):
    text, error = validate_text(text)
    if error:
        return {"md": error}, False
    system = (
        "You are an academic writing tutor. You teach paragraph structure: a topic sentence, support or evidence, "
        "explanation of the evidence, and a closing sentence that links to the argument."
    )
    user = f"""Analyse the structure of this paragraph.

Reply in exactly this format:

### Topic sentence
Verdict: Strong, Weak or Missing. Quote the sentence if there is one, then say in one or two sentences how to improve it.

### Support and evidence
Verdict: Strong, Weak or Missing. Quote the supporting sentences and say what kind of support is missing, if any.

### Explanation
Verdict: Strong, Weak or Missing. Does the writer explain why the evidence matters? Give a short example of what to add.

### Closing link
Verdict: Strong, Weak or Missing. Does the last sentence link back to the main point?

### Unity and flow
Name any sentence that is off-topic, and any place where a transition would help.

### Revised paragraph
Rewrite the paragraph with a clear structure. Do not add facts, numbers or citations that are not in the original.

PARAGRAPH:
\"\"\"{text}\"\"\""""
    md, ok = llm(system, user, temperature=0.3, max_tokens=1800)
    payload = {"md": md}
    if ok:
        payload["revised"] = section(md, "Revised paragraph")
    return payload, ok


# --------------------------------------------------------------------------
# Paraphrase
# --------------------------------------------------------------------------
def paraphrase(text, tone, keep_terms):
    text, error = validate_text(text)
    if error:
        return {"md": error}, False
    system = (
        "You are an academic writing assistant. You paraphrase sentences into clear, formal academic English "
        "while preserving the original meaning. You never add new facts, numbers, or citations."
    )
    terms = f"Keep these key terms unchanged: {keep_terms}." if keep_terms and keep_terms.strip() else ""
    user = f"""Paraphrase the text below. Target tone: {tone}. {terms}

Reply in exactly this format:

### Version 1 (closest to the original)
...

### Version 2 (different sentence structure)
...

### Version 3 (most concise)
...

### What changed
(2-3 bullet points explaining the main techniques used, e.g. synonym use, changing voice, restructuring clauses)

TEXT:
\"\"\"{text}\"\"\""""
    md, ok = llm(system, user, temperature=0.6, max_tokens=1800, use_cache=False)
    if ok:
        md += ("\n\nParaphrasing someone else's idea still needs a citation. "
               "Use this to improve your own wording, not to avoid plagiarism checks.")
    return {"md": md}, ok


# --------------------------------------------------------------------------
# Essay plan
# --------------------------------------------------------------------------
def plan_writing(topic, essay_type, level):
    topic, error = validate_text(topic)
    if error:
        return {"md": error}, False
    system = (
        "You are an academic writing mentor. You help students plan before they write. "
        "You never invent citations, statistics or research findings."
    )
    user = f"""A student wants to write a {essay_type} (level: {level}) about this topic or question:

\"\"\"{topic}\"\"\"

Reply in exactly this format:

### Thesis options
(three different, specific, arguable thesis statements, each in one sentence, in academic English)

### Outline
(Introduction, 3 body paragraphs, Conclusion. For each: a topic sentence and 2 bullet points of ideas to develop.)

### Useful vocabulary
(8-10 academic words or collocations the student can use)

### Mistakes to avoid
(3 short bullet points specific to this kind of writing)

Do not add citations or statistics. Remind the student to find and cite real sources."""
    md, ok = llm(system, user, temperature=0.5, max_tokens=1800)
    return {"md": md}, ok


TOPIC_IDEAS = [
    "Should English be taught from kindergarten in Indonesian schools?",
    "The effect of social media on students' academic writing",
    "Online learning or face-to-face learning: which supports language learners better?",
    "Is memorizing vocabulary lists an effective way to learn English?",
    "The role of motivation in second language acquisition",
    "Should students be allowed to use AI tools for writing assignments?",
    "Does watching English films improve listening skills?",
    "The advantages and disadvantages of group work in the classroom",
    "Is grammar instruction necessary for communicative competence?",
    "How can teachers give feedback that helps students revise their writing?",
]


# --------------------------------------------------------------------------
# APA 7 references (no AI)
# --------------------------------------------------------------------------
def _initials(given):
    out = []
    for token in re.split(r"\s+", given.strip()):
        parts = [p for p in token.split("-") if p]
        if parts:
            out.append("-".join(p[0].upper() + "." for p in parts))
    return " ".join(out)


def _split_name(raw):
    """'Surname, Given names' is a person. A line without a comma is kept as an organization name."""
    raw = raw.strip()
    if "," in raw:
        sur, given = raw.split(",", 1)
        return sur.strip(), given.strip()
    return raw, ""


def apa_author(raw):
    sur, given = _split_name(raw)
    ini = _initials(given) if given else ""
    return f"{sur}, {ini}" if ini else sur


def apa_authors(names):
    a = [apa_author(n) for n in names if n.strip()]
    if not a:
        return ""
    if len(a) == 1:
        return a[0]
    if len(a) <= 20:
        return ", ".join(a[:-1]) + ", & " + a[-1]
    return ", ".join(a[:19]) + ", . . . " + a[-1]


def _end(s):
    s = s.strip()
    return s if s.endswith((".", "?", "!")) else s + "."


def apa_reference(kind, names, year, month_day, title, source, volume, issue, pages, publisher, edition, link):
    names = [n for n in names if n.strip()]
    auth = apa_authors(names)
    date = (year.strip() or "n.d.") + (f", {month_day.strip()}" if month_day.strip() and kind == "Webpage" else "")
    link = link.strip()
    if link and re.match(r"^10\.\d{4,}", link):
        link = "https://doi.org/" + link
    title = title.strip()
    if kind == "Journal article":
        src = f"*{source.strip()}*"
        if volume.strip():
            src += f", *{volume.strip()}*"
        if issue.strip():
            src += f"({issue.strip()})"
        if pages.strip():
            src += f", {pages.strip().replace('-', '–')}"
        if auth:
            ref = f"{_end(auth)} ({date}). {_end(title)} {src}."
        else:
            ref = f"{_end(title)} ({date}). {src}."
    elif kind == "Book":
        ed = f" ({edition.strip()} ed.)" if edition.strip() else ""
        pub = f" {_end(publisher)}" if publisher.strip() else ""
        if auth:
            ref = f"{_end(auth)} ({date}). *{title.rstrip('.')}*{ed}.{pub}"
        else:
            ref = f"*{title.rstrip('.')}*{ed}. ({date}).{pub}"
    else:  # Webpage
        site = source.strip()
        if auth:
            ref = f"{_end(auth)} ({date}). *{title.rstrip('.')}*."
        else:
            ref = f"*{title.rstrip('.')}*. ({date})."
        if site and site != auth:
            ref += f" {_end(site)}"
    ref = ref.strip()
    if link:
        ref += f" {link}"
    return ref


def apa_in_text(names, year):
    sur = [_split_name(n)[0] for n in names if n.strip()]
    y = year.strip() or "n.d."
    if not sur:
        return "", ""
    if len(sur) == 1:
        return f"({sur[0]}, {y})", f"{sur[0]} ({y})"
    if len(sur) == 2:
        return f"({sur[0]} & {sur[1]}, {y})", f"{sur[0]} and {sur[1]} ({y})"
    return f"({sur[0]} et al., {y})", f"{sur[0]} et al. ({y})"


# --------------------------------------------------------------------------
# Text insights (no AI)
# --------------------------------------------------------------------------
STOPWORDS = set(
    "a an the and or but if of to in on at for with by from as is are was were be been being it its this that "
    "these those he she they we you i not no so than then there their them his her our your which who whom "
    "what when where while also can could may might will would should have has had do does did more most such "
    "into about over between through during per".split()
)
TRANSITIONS = [
    "however", "moreover", "furthermore", "therefore", "thus", "consequently", "in addition", "for example",
    "for instance", "in contrast", "on the other hand", "nevertheless", "as a result", "in conclusion",
    "firstly", "secondly", "finally", "similarly", "although", "whereas", "in particular", "overall",
]
PASSIVE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?"
    r"(?:\w+(?:ed|en)|taught|shown|known|made|done|found|built|held|kept|led|sent|told|thought|brought|bought|caught|understood)\b",
    re.I)


def count_syllables(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def text_insights(text):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    lower = [w.lower() for w in words]
    n_words, n_sent = len(words), max(len(sentences), 1)
    syllables = sum(count_syllables(w) for w in words)
    flesch = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / max(n_words, 1))
    freq = {}
    for w in lower:
        if w not in STOPWORDS and len(w) > 3:
            freq[w] = freq.get(w, 0) + 1
    repeated = sorted(((w, c) for w, c in freq.items() if c >= 3), key=lambda x: -x[1])[:8]
    low_text = text.lower()
    return {
        "words": n_words, "sentences": len(sentences),
        "avg_sentence": n_words / n_sent,
        "variety": len(set(lower)) / max(n_words, 1),
        "flesch": flesch,
        "lengths": [len(s.split()) for s in sentences],
        "long_sentences": [s for s in sentences if len(s.split()) > 30],
        "repeated": repeated,
        "transitions": [t for t in TRANSITIONS if re.search(rf"\b{re.escape(t)}\b", low_text)],
        "passive": [m.group(0) for m in PASSIVE.finditer(text)],
    }


def flesch_label(score):
    if score >= 70:
        return "easy to read"
    if score >= 50:
        return "fairly difficult"
    if score >= 30:
        return "difficult, which is normal for academic writing"
    return "very difficult"


# --------------------------------------------------------------------------
# Phrasebank (no AI)
# --------------------------------------------------------------------------
PHRASEBANK = {
    "Introducing a topic": [
        "In recent years, there has been growing interest in ...",
        "... has become an increasingly important issue in ...",
        "One of the most widely discussed topics in ... is ...",
        "The aim of this paper is to examine ...",
        "This essay argues that ...",
        "This study seeks to explore the relationship between ... and ...",
    ],
    "Giving background": [
        "Previous research has shown that ...",
        "It is widely acknowledged that ...",
        "According to [Author, Year], ...",
        "A number of researchers have suggested that ...",
        "Traditionally, ... has been regarded as ...",
        "Despite extensive research, little is known about ...",
    ],
    "Hedging a claim": [
        "This may suggest that ...",
        "It appears that ...",
        "The results seem to indicate that ...",
        "This could be attributed to ...",
        "To some extent, ...",
        "It is possible that ...",
    ],
    "Adding and listing ideas": [
        "In addition to ..., ...",
        "Furthermore, ...",
        "Moreover, ...",
        "Another important factor is ...",
        "First and foremost, ... Secondly, ... Finally, ...",
        "Not only ... but also ...",
    ],
    "Contrasting and conceding": [
        "However, it should be noted that ...",
        "On the other hand, ...",
        "In contrast to ..., ...",
        "Although ..., it is also true that ...",
        "While some argue that ..., others claim that ...",
        "Nevertheless, ...",
    ],
    "Cause and effect": [
        "As a result of ..., ...",
        "This leads to ...",
        "One consequence of ... is ...",
        "... can be explained by the fact that ...",
        "Owing to ..., ...",
        "This is largely due to ...",
    ],
    "Giving examples": [
        "For example, ...",
        "For instance, ...",
        "... is a case in point.",
        "This is illustrated by ...",
        "A typical example of this is ...",
        "To illustrate this point, ...",
    ],
    "Describing data": [
        "As shown in Table 1, ...",
        "Figure 2 illustrates that ...",
        "There was a significant increase in ... between ... and ...",
        "The data reveal that ...",
        "Approximately X% of the participants ...",
        "The majority of respondents ...",
    ],
    "Evaluating ideas and sources": [
        "This argument is convincing because ...",
        "One limitation of this study is ...",
        "The evidence presented is insufficient to ...",
        "A key strength of this approach is ...",
        "This view has been challenged by ...",
        "These findings are consistent with ...",
    ],
    "Concluding": [
        "In conclusion, ...",
        "To sum up, ...",
        "Overall, the evidence suggests that ...",
        "The findings of this study highlight the importance of ...",
        "Further research is needed to ...",
        "These results have important implications for ...",
    ],
}


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
SAMPLE = (
    "Nowadays, many students dont know how to write a good essay. "
    "They get a lot of problems when they writes the introduction, and I think this is a big thing."
)
SAMPLE_PARAGRAPH = (
    "Social media is used by most students. Many students check it every hour. "
    "I like pizza too. Because of this, social media changes how students study."
)
GENRES = ["academic paragraph", "academic essay", "abstract", "argumentative essay", "report"]
TONES = ["Formal academic", "Neutral academic", "Concise and direct"]
ESSAY_TYPES = ["argumentative essay", "opinion essay", "cause-and-effect essay", "compare-and-contrast essay",
               "problem-solution essay", "research paper introduction"]
LEVELS = ["beginner", "intermediate", "advanced"]
RATING = [1, 2, 3, 4, 5]

TOOLS = ["Grammar", "Style", "Rubric", "Paragraph", "Paraphrase", "Plan", "APA 7", "Insights", "Phrases", "Feedback"]
DESCRIPTIONS = {
    "Grammar": ("Grammar check",
                "Paste a paragraph. You get the corrected text with every change marked, and the rule behind each one."),
    "Style": ("Academic style",
              "Finds contractions, vague words and informal phrases, then rewrites the text in a formal register."),
    "Rubric": ("Rubric feedback",
               "Scores your writing from 1 to 5 on five criteria and tells you what to fix first. "
               "Run it again after each revision to see your progress."),
    "Paragraph": ("Paragraph check",
                  "Checks for a topic sentence, support, explanation and a closing link, and shows what is missing."),
    "Paraphrase": ("Paraphrase",
                   "Three ways to say the same thing in academic English. "
                   "Cite the source whenever the idea is not your own."),
    "Plan": ("Essay plan",
             "Enter a topic or question and get thesis options, an outline and useful vocabulary before you write."),
    "APA 7": ("APA 7 references",
              "Fill in the details of a source to get a formatted reference and in-text citations. "
              "AI is not used here, so nothing is invented."),
    "Insights": ("Text insights",
                 "Statistics and sentence rhythm for your draft. This tool works without AI, so it is always available."),
    "Phrases": ("Academic phrases",
                "Sentence starters for common academic moves. Replace the dots with your own ideas."),
    "Feedback": ("Your feedback",
                 "Tell us how Wello worked for you. Your answers are used for research."),
}
FEATURES = ["Grammar check", "Academic style", "Rubric feedback", "Paragraph check", "Paraphrase",
            "Essay plan", "APA 7 references", "Text insights", "Academic phrases", "More than one tool"]


def set_text(key, value):
    st.session_state[key] = value


def pick_topic():
    st.session_state["t_plan"] = random.choice(TOPIC_IDEAS)


def word_count(text):
    return len((text or "").split())


SHARED = "draft_w"  # widget key of the one draft that Grammar, Style, Rubric, Paragraph and Insights share


def apply_draft(new_text):
    """Put a corrected or revised text into the draft. The old draft is kept for Undo."""
    st.session_state["draft_prev"] = st.session_state.get(SHARED, st.session_state.get("draft", ""))
    st.session_state[SHARED] = new_text


def undo_draft():
    prev = st.session_state.get("draft_prev")
    if prev is not None:
        st.session_state[SHARED] = prev
        st.session_state["draft_prev"] = None


def add_phrase(phrase):
    draft = st.session_state.get("draft", "")
    joiner = "" if not draft or draft.endswith((" ", "\n")) else " "
    st.session_state["draft"] = draft + joiner + phrase
    st.session_state.pop(SHARED, None)  # the editor is not on screen; it rebuilds from "draft"
    st.toast("Added to your draft.")


def writing_surface(key, sample=None, height=230, placeholder="Paste or type your text here.", shared=False):
    """The ruled writing page. With shared=True the text is the single draft used by several tools."""
    wkey = SHARED if shared else key
    if wkey not in st.session_state:
        st.session_state[wkey] = st.session_state.get("draft", "") if shared else ""
    with st.container(key=f"paper_{key}"):
        text = st.text_area("Your text", key=wkey, height=height, placeholder=placeholder,
                            label_visibility="collapsed")
    if shared:
        st.session_state["draft"] = text
    can_undo = shared and st.session_state.get("draft_prev") is not None
    cols = st.columns([3, 1.1, 1, 1] if can_undo else [3, 1.1, 1])
    cols[0].markdown(f'<div class="counter">{word_count(text):,} of {MAX_WORDS:,} words</div>', unsafe_allow_html=True)
    cols[1].button("Use example", key=f"ex_{key}", on_click=set_text, args=(wkey, sample),
                   use_container_width=True, disabled=sample is None)
    cols[2].button("Clear", key=f"clr_{key}", on_click=set_text, args=(wkey, ""), use_container_width=True)
    if can_undo:
        cols[3].button("Undo", key=f"undo_{key}", on_click=undo_draft, use_container_width=True)
    if shared:
        st.markdown('<p class="note">This draft stays here when you switch between Grammar, Style, Rubric, '
                    'Paragraph and Insights.</p>', unsafe_allow_html=True)
    return text


def section(md, heading):
    """Text under '### heading' in a markdown answer, or ''. """
    m = re.search(rf"###\s*{re.escape(heading)}[^\n]*\n(.*?)(?=\n###|\Z)", md or "", re.S | re.I)
    return m.group(1).strip().strip("\"“”") if m else ""


def use_in_draft_button(new_text, label, key):
    """Button that replaces the draft with new_text, or a note when that already happened."""
    if not new_text:
        return
    if st.session_state.get("draft", "").strip() == new_text.strip():
        st.caption("This version is now in your draft.")
    else:
        st.button(label, key=key, on_click=apply_draft, args=(new_text,))


def chips(counts):
    return "".join(f'<span class="chip">{html.escape(k)} <b>{v}</b></span>' for k, v in counts.items())


def run_tool(result_key, feature, spinner, fn, *args, words=0):
    """Run a tool, keep its result in session state, and log anonymous usage."""
    if time.time() - st.session_state.get("last_call", 0) < COOLDOWN_SECONDS:
        st.warning("Wait a few seconds before the next request.")
        return None
    st.session_state["last_call"] = time.time()
    with st.spinner(spinner):
        payload, ok = fn(*args)
    st.session_state[result_key] = payload
    log_usage(feature, words, "ok" if ok else "error")
    return payload


def show_sheet(result_key, md, filename):
    with st.container(key=f"sheet_{result_key}"):
        st.markdown(md)
    st.download_button("Download as text", md, file_name=filename, key=f"dl_{result_key}")


def tool_header(tool):
    title, desc = DESCRIPTIONS[tool]
    st.markdown(f'<div class="tool-title">{title}</div><div class="tool-desc">{desc}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Masthead and navigation
# --------------------------------------------------------------------------
st.markdown(
    f'<div class="masthead">{logo_img()}<div><div class="name">Wello</div>'
    '<div class="line">Check, rewrite and plan your academic writing in English.</div></div></div>',
    unsafe_allow_html=True,
)

if not GROQ_API_KEY:
    st.info("The AI key is not set. Add GROQ_API_KEY in Secrets to use the tools that need AI. "
            "APA 7, Insights and Phrases work without it.")

choice = st.pills("Tool", TOOLS, selection_mode="single", default=TOOLS[0], key="tool_pills",
                  label_visibility="collapsed")
tool = choice or st.session_state.get("last_tool", TOOLS[0])
st.session_state["last_tool"] = tool
tool_header(tool)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
if tool == "Grammar":
    text = writing_surface("draft", sample=SAMPLE, shared=True)
    if st.button("Check grammar", type="primary", key="go_grammar"):
        payload = run_tool("r_grammar", "grammar", "Reading your text...", check_grammar, text,
                           words=word_count(text))
        if payload and payload.get("tags"):
            profile = st.session_state.setdefault("error_profile", {})
            for tag, n in payload["tags"].items():
                profile[tag] = profile.get(tag, 0) + n
    res = st.session_state.get("r_grammar")
    if res:
        if res.get("corrected") and res.get("original"):
            body, n = tracked_changes_html(res["original"], res["corrected"])
            if res["original"].strip() != st.session_state.get("draft", "").strip() \
                    and res["corrected"].strip() != st.session_state.get("draft", "").strip():
                st.caption("This feedback is for an earlier version of your draft.")
            st.markdown(f"**{n} change{'s' if n != 1 else ''}**" if n else "**No changes needed**")
            st.markdown(f'<div class="track">{body}</div>'
                        '<div class="legend">Red with a line through it was removed. Green with an underline was added.</div>',
                        unsafe_allow_html=True)
            if res.get("tags"):
                st.markdown('<div class="eyebrow">Kinds of mistakes in this text</div>' + chips(res["tags"]),
                            unsafe_allow_html=True)
            profile = st.session_state.get("error_profile", {})
            if profile and sum(profile.values()) > sum((res.get("tags") or {}).values()):
                top = dict(sorted(profile.items(), key=lambda kv: -kv[1])[:6])
                st.markdown('<div class="eyebrow">Most frequent in this session</div>' + chips(top),
                            unsafe_allow_html=True)
            st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
            use_in_draft_button(res["corrected"], "Use the corrected text as my draft", "use_corrected")
            st.caption("Corrected text, ready to copy")
            st.code(res["corrected"], language=None, wrap_lines=True)
            rest = TAG_RE.sub(lambda m: f"**{m.group(1)}:**", CORRECTED_RE.sub("", res["md"]).strip())
            if rest:
                with st.container(key="sheet_grammar_notes"):
                    st.markdown(rest)
            st.download_button("Download corrected text", res["corrected"], file_name="corrected_text.txt",
                               key="dl_grammar")
        else:
            show_sheet("grammar", res["md"], "grammar_result.txt")

elif tool == "Style":
    text = writing_surface("draft", sample=SAMPLE, shared=True)
    if st.button("Check style", type="primary", key="go_style"):
        run_tool("r_style", "style", "Reading your text...", check_style, text, words=word_count(text))
    res = st.session_state.get("r_style")
    if res:
        show_sheet("style", res["md"], "style_result.txt")
        use_in_draft_button(res.get("revised"), "Use the revised version as my draft", "use_style")

elif tool == "Rubric":
    c1, c2 = st.columns([1, 2])
    genre = c1.selectbox("Type of writing", GENRES, key="genre_rubric")
    task = c2.text_input("Writing task (optional)", key="task_rubric",
                         placeholder="e.g. Discuss the advantages and disadvantages of online learning.")
    text = writing_surface("draft", sample=SAMPLE, height=300, placeholder="Paste your paragraph or essay here.",
                           shared=True)
    if st.button("Score my writing", type="primary", key="go_rubric"):
        payload = run_tool("r_rubric", "rubric", "Scoring your writing...", rubric_feedback, text, genre, task,
                           words=word_count(text))
        if payload and payload.get("rubric"):
            st.session_state.setdefault("drafts", []).append(
                {"total": sum(payload["rubric"]["scores"]), "words": payload["rubric"]["words"]})
    res = st.session_state.get("r_rubric")
    if res:
        r = res.get("rubric")
        if not r:
            show_sheet("rubric", res["md"], "rubric_feedback.txt")
        else:
            total = sum(r["scores"])
            st.markdown(f'<div class="total">{total}<small> / 25</small></div>'
                        f'<div class="tool-desc" style="margin-top:.3rem">{rubric_level(total)}</div>',
                        unsafe_allow_html=True)
            rows = []
            for name, score, cmt in zip(CRITERIA, r["scores"], r["comments"]):
                tally = "".join(f'<i class="{"on" if i < score else ""}"></i>' for i in range(5))
                rows.append(
                    f'<div class="crit"><div class="row"><span class="cname">{html.escape(name)}</span>'
                    f'<span><span class="tally">{tally}</span><span class="score">{score}</span></span></div>'
                    f'<div class="cmt">{html.escape(cmt)}</div></div>')
            st.markdown("".join(rows) + '<div style="height:1rem"></div>', unsafe_allow_html=True)
            md = ""
            for title, key in (("Strengths", "strengths"), ("What to improve", "improve"), ("Next steps", "next_steps")):
                if r[key]:
                    md += f"### {title}\n" + "\n".join(f"- {x}" for x in r[key]) + "\n\n"
            with st.container(key="sheet_rubric_notes"):
                st.markdown(md)
            st.download_button("Download as text", rubric_as_text(r), file_name="rubric_feedback.txt", key="dl_rubric")
    drafts = st.session_state.get("drafts", [])
    if len(drafts) > 1:
        st.markdown("**Your drafts**")
        first = drafts[0]["total"]
        for i, d in enumerate(drafts, 1):
            delta = "" if i == 1 else f", {d['total'] - first:+d} since draft 1"
            st.markdown(f"Draft {i}: {d['total']} / 25, {d['words']} words{delta}")
    if drafts:
        st.button("Start a new essay", key="reset_drafts",
                  on_click=lambda: (st.session_state.__setitem__("drafts", []),
                                    st.session_state.__setitem__("r_rubric", None)))

elif tool == "Paragraph":
    text = writing_surface("draft", sample=SAMPLE_PARAGRAPH, height=200,
                           placeholder="Paste one paragraph here.", shared=True)
    if st.button("Check paragraph", type="primary", key="go_para"):
        run_tool("r_para", "paragraph", "Reading your paragraph...", check_paragraph, text, words=word_count(text))
    res = st.session_state.get("r_para")
    if res:
        show_sheet("para", res["md"], "paragraph_check.txt")
        use_in_draft_button(res.get("revised"), "Use the revised paragraph as my draft", "use_para")

elif tool == "Paraphrase":
    text = writing_surface("t_para2", height=180, placeholder="Paste the sentence or sentences to paraphrase.")
    if st.session_state.get("draft", "").strip():
        st.button("Copy my draft into this box", key="draft_to_para", on_click=set_text,
                  args=("t_para2", st.session_state["draft"]))
    c1, c2 = st.columns(2)
    tone = c1.selectbox("Tone", TONES, key="tone_p")
    terms = c2.text_input("Key terms to keep (optional)", key="terms_p", placeholder="e.g. motivation, second language")
    if st.button("Paraphrase", type="primary", key="go_p"):
        run_tool("r_p", "paraphrase", "Rewriting...", paraphrase, text, tone, terms, words=word_count(text))
    res = st.session_state.get("r_p")
    if res:
        show_sheet("p", res["md"], "paraphrase_result.txt")

elif tool == "Plan":
    if "t_plan" not in st.session_state:
        st.session_state["t_plan"] = ""
    with st.container(key="paper_t_plan"):
        topic = st.text_area("Topic", key="t_plan", height=130, label_visibility="collapsed",
                             placeholder="Type a topic or question, or press Suggest a topic.")
    c1, c2, c3 = st.columns([1.3, 1.3, 1])
    essay_type = c1.selectbox("Type of writing", ESSAY_TYPES, key="et_plan")
    level = c2.selectbox("Your level", LEVELS, index=1, key="lv_plan")
    c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    c3.button("Suggest a topic", key="suggest_topic", on_click=pick_topic, use_container_width=True)
    if st.button("Make a plan", type="primary", key="go_plan"):
        run_tool("r_plan", "plan", "Planning...", plan_writing, topic, essay_type, level, words=word_count(topic))
    res = st.session_state.get("r_plan")
    if res:
        show_sheet("plan", res["md"], "essay_plan.txt")

elif tool == "APA 7":
    kind = st.radio("Source type", ["Journal article", "Book", "Webpage"], horizontal=True, key="apa_kind")
    st.markdown('<p class="note">Write titles in sentence case, as APA asks: capitalize only the first word, '
                'the first word after a colon, and proper nouns.</p>', unsafe_allow_html=True)
    with st.form("apa_form"):
        authors = st.text_area("Authors, one per line", height=100, placeholder="Smith, John A.\nNguyen, Lan",
                               help="Use Surname, Given names. For an organization, write its name on one line.")
        c1, c2 = st.columns(2)
        year = c1.text_input("Year", placeholder="2021")
        month_day = c2.text_input("Month and day (webpages only)", placeholder="March 5")
        title = st.text_input("Title", placeholder="Corrective feedback in second language writing")
        source = pages = volume = issue = publisher = edition = ""
        if kind == "Journal article":
            source = st.text_input("Journal name", placeholder="Journal of Second Language Writing")
            c1, c2, c3 = st.columns(3)
            volume = c1.text_input("Volume", placeholder="52")
            issue = c2.text_input("Issue", placeholder="3")
            pages = c3.text_input("Pages", placeholder="101-118")
        elif kind == "Book":
            c1, c2 = st.columns([2, 1])
            publisher = c1.text_input("Publisher", placeholder="Oxford University Press")
            edition = c2.text_input("Edition (optional)", placeholder="2nd")
        else:
            source = st.text_input("Website name", placeholder="British Council")
        link = st.text_input("DOI or URL (optional)", placeholder="10.1016/j.jslw.2021.100800")
        made = st.form_submit_button("Format reference", type="primary")
    if made:
        names = [n for n in authors.splitlines() if n.strip()]
        no_comma = [n.strip() for n in names if "," not in n]
        if no_comma:
            st.info("No comma in: " + "; ".join(no_comma) + ". Wello treats this as an organization name. "
                    "For a person, write Surname, Given names.")
        if not title.strip():
            st.warning("Add the title of the source.")
        elif kind == "Journal article" and not source.strip():
            st.warning("Add the journal name.")
        else:
            ref = apa_reference(kind, names, year, month_day, title, source, volume, issue, pages, publisher,
                                edition, link)
            paren, narr = apa_in_text(names, year)
            st.session_state["apa_last"] = {"ref": ref, "paren": paren, "narr": narr}
    last = st.session_state.get("apa_last")
    if last:
        st.markdown("**Reference**")
        with st.container(key="sheet_apa"):
            st.markdown(last["ref"])
        st.markdown('<p class="note">Select the reference and copy it. Italics stay when you paste into Word.</p>',
                    unsafe_allow_html=True)
        if last["paren"]:
            st.markdown(f"**In-text:** {last['paren']} or {last['narr']}")
        if st.button("Add to my reference list", key="apa_add"):
            st.session_state.setdefault("refs", []).append(last["ref"])
    refs = st.session_state.get("refs", [])
    if refs:
        st.markdown(f"**My reference list** ({len(refs)})")
        ordered = sorted(refs, key=lambda r: re.sub(r"[^a-z]", "", r.lower()))
        with st.container(key="sheet_reflist"):
            for r in ordered:
                st.markdown(r)
        plain = "\n\n".join(r.replace("*", "") for r in ordered)
        c1, c2 = st.columns([1, 1])
        c1.download_button("Download list as text", plain, file_name="references.txt", key="dl_refs")
        c2.button("Clear list", key="clear_refs", on_click=lambda: st.session_state.__setitem__("refs", []))

elif tool == "Insights":
    text = writing_surface("draft", sample=SAMPLE, shared=True)
    info = None
    if text.strip():
        clean, err = validate_text(text)
        if err:
            st.warning(err)
        else:
            info = text_insights(clean)  # updates as you type, no button needed
    else:
        st.markdown('<p class="note">Write or paste a draft above and the numbers appear here.</p>',
                    unsafe_allow_html=True)
    if info:
        st.markdown(
            '<div class="stats">'
            f'<div><b>{info["words"]}</b><span>words</span></div>'
            f'<div><b>{info["sentences"]}</b><span>sentences</span></div>'
            f'<div><b>{info["avg_sentence"]:.1f}</b><span>words per sentence</span></div>'
            f'<div><b>{info["variety"]:.0%}</b><span>different words</span></div>'
            f'<div><b>{info["flesch"]:.0f}</b><span>reading ease</span></div>'
            f'<div><b>{len(info["passive"])}</b><span>possible passives</span></div>'
            '</div>', unsafe_allow_html=True)
        st.markdown(f"The text is **{flesch_label(info['flesch'])}**. Academic writing usually scores between 20 and 50.")
        if len(info["lengths"]) > 1:
            st.markdown("**Sentence rhythm** (words in each sentence, in order)")
            import pandas as pd

            rhythm = pd.DataFrame({"Words": info["lengths"]}, index=range(1, len(info["lengths"]) + 1))
            st.bar_chart(rhythm, color=ACCENT, height=170, x_label="Sentence number", y_label="Words")
            st.markdown('<p class="note">A mix of short and long sentences reads better than a row of equal ones.</p>',
                        unsafe_allow_html=True)
        with st.container(key="sheet_insights"):
            st.markdown("**Transitions used:** " + (", ".join(info["transitions"]) if info["transitions"]
                        else "none found. Try *however*, *moreover* or *therefore*."))
            st.markdown("**Words repeated three times or more:** "
                        + (", ".join(f"{w} ({c})" for w, c in info["repeated"]) or "none."))
            if info["long_sentences"]:
                st.markdown(f"**Sentences over 30 words ({len(info['long_sentences'])}).** Consider splitting:")
                for s in info["long_sentences"][:5]:
                    st.markdown(f"> {s}")
            else:
                st.markdown("**Long sentences:** none over 30 words.")
            if info["passive"]:
                st.markdown("**Possible passive structures:** " + "; ".join(f"*{p}*" for p in info["passive"][:8]))
        st.markdown('<p class="note">These numbers come from simple rules. Use them as a guide, not as a grade.</p>',
                    unsafe_allow_html=True)

elif tool == "Phrases":
    c1, c2 = st.columns(2)
    category = c1.selectbox("Category", ["All"] + list(PHRASEBANK.keys()), key="pb_cat")
    query = c2.text_input("Search", key="pb_q", placeholder="e.g. however")
    shown, listed = 0, []
    with st.container(key="sheet_phrases"):
        for cat, phrases in PHRASEBANK.items():
            if category != "All" and cat != category:
                continue
            matching = [p for p in phrases if query.strip().lower() in p.lower()]
            if matching:
                st.markdown(f"### {cat}")
                st.markdown("\n".join(f"- {p}" for p in matching))
                shown += len(matching)
                listed += matching
        if shown == 0:
            st.markdown("No phrase matches. Try another word or choose All.")
    if listed:
        c1, c2 = st.columns([4, 1])
        pick = c1.selectbox("Add a phrase to my draft", listed, index=None, key="pb_pick",
                            placeholder="Choose a phrase")
        c2.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        c2.button("Add", key="pb_add", disabled=pick is None, use_container_width=True,
                  on_click=add_phrase, args=(pick,) if pick else ("",))
        n_draft = word_count(st.session_state.get("draft", ""))
        st.markdown(f'<p class="note">Your draft has {n_draft:,} words. Open Grammar, Style or Rubric to see it. '
                    'Complete a phrase with your own ideas, and do not copy whole sentences from other authors.</p>',
                    unsafe_allow_html=True)

elif tool == "Feedback":
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name (optional)")
        c1, c2 = st.columns(2)
        role = c1.selectbox("You are a", ["Student", "Lecturer or teacher", "Other"])
        feature = c2.selectbox("Which tool did you use?", FEATURES)
        ease = st.radio("Ease of use (1 = very hard, 5 = very easy)", RATING, index=None, horizontal=True)
        useful = st.radio("Usefulness (1 = not useful, 5 = very useful)", RATING, index=None, horizontal=True)
        accuracy = st.radio("Accuracy of the feedback (1 = poor, 5 = excellent)", RATING, index=None, horizontal=True)
        design = st.radio("Look and layout (1 = poor, 5 = excellent)", RATING, index=None, horizontal=True)
        comment = st.text_area("Comments or suggestions", height=100)
        submitted = st.form_submit_button("Send feedback", type="primary")

    if submitted:
        if None in (ease, useful, accuracy, design):
            st.warning("Rate all four items before sending.")
        else:
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "name": (name or "").strip() or "Anonymous",
                "role": role, "feature_used": feature,
                "ease_of_use": ease, "usefulness": useful, "accuracy": accuracy, "design": design,
                "comment": (comment or "").strip(),
            }
            append_csv(FEEDBACK_FILE, FEEDBACK_FIELDS, row)
            push_to_dataset(FEEDBACK_FILE, FEEDBACK_FIELDS, row)
            st.success("Feedback sent. Thank you.")

    with st.expander("Admin: download collected data"):
        pw = st.text_input("Admin password", type="password", key="admin_pw")
        if pw:
            if ADMIN_PASSWORD and pw == ADMIN_PASSWORD:
                for fname, label in ((FEEDBACK_FILE, "feedback"), (USAGE_FILE, "usage log")):
                    if os.path.exists(fname):
                        with open(fname, "rb") as f:
                            st.download_button(f"Download {label} ({fname})", f.read(), file_name=fname,
                                               mime="text/csv", key=f"dl_admin_{label}")
                    else:
                        st.info(f"No {label} data yet.")
            else:
                st.error("Wrong password, or ADMIN_PASSWORD is not set.")


# --------------------------------------------------------------------------
# About and privacy
# --------------------------------------------------------------------------
st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
with st.expander("About Wello and your privacy"):
    st.markdown(
        f"Wello uses open-source language models (currently `{MODEL_ID}`) to give feedback on academic writing. "
        f"Each request can have up to {MAX_WORDS:,} words. The AI can make mistakes, so check important corrections yourself.\n\n"
        "Your text is sent to a third-party AI service (Groq) only to produce feedback, and Wello does not store it. "
        "For research, Wello records which tool was used and how many words, never the text itself. "
        "The feedback form is stored only if you send it.\n\n"
        "When the free usage limit is reached, wait a minute and try again."
    )
