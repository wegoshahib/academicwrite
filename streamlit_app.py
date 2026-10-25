"""
Wello v2 - an AI assistant for academic writing in English.
Built with Streamlit. Runs for free on Streamlit Community Cloud.

Tools
1. Grammar & Spelling Checker      (AI + LanguageTool)
2. Academic Style Checker          (rules + AI)
3. Rubric-Based Feedback           (AI)
4. Academic Paraphraser            (AI)
5. Writing Planner                 (AI)  - thesis statements and outline
6. Text Insights                   (no AI) - readability and statistics
7. Academic Phrasebank             (no AI) - ready-to-use phrases
8. Feedback form                   (research data)
"""

import csv
import os
import re
import time
from datetime import datetime, timezone

import base64

import requests
import streamlit as st

LOGO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
HAS_LOGO = os.path.exists(LOGO_FILE)

st.set_page_config(
    page_title="Wello",
    page_icon=LOGO_FILE if HAS_LOGO else "✍️",
    layout="wide",
)


def logo_html(size=64):
    """Inline logo for the header. Falls back to an emoji if logo.png is missing."""
    if not HAS_LOGO:
        return '<span style="font-size:2.4rem">✍️</span>'
    with open(LOGO_FILE, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return (f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}" '
            'style="border-radius:16px; display:block;" alt="Wello logo">')


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
HF_TOKEN = clean_secret(get_secret("HF_TOKEN", ""))  # optional: save data to a private HF dataset
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


# --------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 3.4rem; max-width: 1050px;}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background-color: #c2620f; border-color: #c2620f; color: #ffffff;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
    background-color: #9a4a0b; border-color: #9a4a0b; color: #ffffff;
}
.stTabs [aria-selected="true"] {color: #c2620f;}
.stTabs [data-baseweb="tab-highlight"] {background-color: #c2620f;}
.hero {
    background: linear-gradient(135deg, #8a3f08 0%, #b45309 50%, #c2620f 100%);
    padding: 1.3rem 1.8rem; border-radius: 18px; margin-bottom: 1.1rem;
    display: flex; align-items: center; gap: 1.1rem;
}
.hero h1 {margin: 0; font-size: 2.1rem; color: #ffffff !important; padding: 0;}
.hero p {margin: .3rem 0 0 0; color: #fff4e8; font-size: 1.02rem;}
.stTabs [data-baseweb="tab-list"] {gap: 4px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 14px; height: auto;}
.stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 600;}
div[data-testid="stMetric"] {
    background: rgba(127, 127, 127, 0.08); border-radius: 12px; padding: .7rem .9rem;
}
.tip {
    background: rgba(210, 113, 27, 0.12); border-left: 4px solid #d2711b;
    padding: .6rem .9rem; border-radius: 8px; font-size: .93rem;
}
.phrase {
    background: rgba(127, 127, 127, 0.08); border-radius: 10px;
    padding: .55rem .85rem; margin-bottom: .4rem; font-size: .98rem;
}
</style>
""",
    unsafe_allow_html=True,
)


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
        api.upload_file(
            path_or_fileobj=tmp, path_in_repo=filename,
            repo_id=FEEDBACK_DATASET, repo_type="dataset",
        )
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
    """Anonymous statistics only: feature name, word count and result. The text itself is never stored."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature": feature, "words": words, "status": status,
    }
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


def _post_groq(model, system_prompt, user_prompt, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens + 1500,
    }
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"  # faster answers, saves free quota
    try:
        r = requests.post(
            GROQ_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=payload, timeout=90,
        )
    except requests.RequestException as exc:
        raise AIError(None, f"{type(exc).__name__}: {exc}") from exc
    if r.status_code != 200:
        raise AIError(r.status_code, r.text[:400])
    content = (r.json()["choices"][0]["message"].get("content") or "").strip()
    if not content:
        raise AIError(None, "The model returned an empty answer.")
    return content


@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def _cached_post(model, system_prompt, user_prompt, temperature, max_tokens):
    return _post_groq(model, system_prompt, user_prompt, temperature, max_tokens)


HINTS = {
    400: "The request was rejected. The model name in MODEL_ID may be wrong.",
    401: "The API key is missing or invalid. Check GROQ_API_KEY in Secrets.",
    403: "Access denied. Check your Groq account and API key.",
    404: "The model is not available. Set MODEL_ID in Secrets to another model.",
    413: "The text is too long for the free limit. Please use a shorter text.",
    429: "The free usage limit is reached. Please wait a minute and try again.",
}


def llm(system_prompt, user_prompt, temperature=0.3, max_tokens=1800, use_cache=True):
    """Call the AI. Returns (text, ok). Tries a fallback model if the main one is busy."""
    call = _cached_post if use_cache else _post_groq
    models = [MODEL_ID] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL_ID else [])
    last_error = None
    for model in models:
        try:
            return call(model, system_prompt, user_prompt, temperature, max_tokens), True
        except AIError as exc:
            last_error = exc
            if exc.status in (401, 403):  # a different model will not help
                break
    detail = last_error.detail.replace(GROQ_API_KEY, "***") if GROQ_API_KEY else last_error.detail
    message = (
        "⚠️ The AI service is busy or unavailable right now. Please wait a moment and try again.\n\n"
        f"{HINTS.get(last_error.status, '')}\n\n"
        f"(status {last_error.status}, key length {len(GROQ_API_KEY)})\n\n"
        f"```\n{detail[:300]}\n```"
    )
    return message, False


def validate_text(text):
    text = (text or "").strip()
    if not text:
        return None, "⚠️ Please paste or type your text first."
    if len(text.split()) > MAX_WORDS:
        return None, f"⚠️ Your text is too long. Please keep it under {MAX_WORDS} words."
    return text, None


# --------------------------------------------------------------------------
# Tool 1: Grammar & Spelling
# --------------------------------------------------------------------------
def languagetool_check(text):
    try:
        r = requests.post(LANGUAGETOOL_URL, data={"text": text, "language": "en-US"}, timeout=20)
        r.raise_for_status()
        return r.json().get("matches", [])
    except Exception:  # noqa: BLE001
        return None


def check_grammar(text):
    text, error = validate_text(text)
    if error:
        return error, False

    matches = languagetool_check(text)
    detected = ""
    if matches:
        lines = []
        for m in matches[:40]:
            start, length = m["offset"], m["length"]
            original = text[start:start + length]
            suggestions = ", ".join(s["value"] for s in m.get("replacements", [])[:3]) or "-"
            lines.append(f"- **{original}** → {suggestions} ({m.get('message', '')})")
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
(the full corrected text)

### Explanation of Corrections
(a numbered list. For each: ~~wrong~~ → **correct**, the grammar rule name, and a one-sentence simple explanation)

### Quick Tip
(one short tip to avoid the most frequent mistake)

If the text has no errors, say so and give one suggestion for improvement.

TEXT:
\"\"\"{text}\"\"\""""
    return llm(system, user, temperature=0.2)


# --------------------------------------------------------------------------
# Tool 2: Academic Style
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
        findings.append(f"- **Contractions** ({', '.join(contractions[:10])}): write them in full form (e.g. *do not*, *it is*).")
    for pattern, suggestion in INFORMAL_PATTERNS.items():
        found = sorted(set(m.group(0).lower() for m in re.finditer(pattern, text, flags=re.I)))
        if found:
            findings.append(f"- **{found[0]}**: consider → {suggestion}")
    personal = sorted(set(m.group(0) for m in re.finditer(PERSONAL_PATTERN, text)))
    if personal:
        findings.append(
            f"- **Personal pronouns** ({', '.join(personal[:8])}): many academic genres prefer an impersonal style "
            "(e.g. *This study argues...* instead of *I think...*). Check your lecturer's guideline."
        )
    return "\n".join(findings)


def check_style(text):
    text, error = validate_text(text)
    if error:
        return error, False

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
    header = "### 🔎 Automatic Scan Results\n" + (scan if scan else "No obvious informal expressions found.") + "\n\n---\n\n"
    return header + ai_part, ok


# --------------------------------------------------------------------------
# Tool 3: Rubric feedback
# --------------------------------------------------------------------------
def rubric_feedback(text, genre, task):
    text, error = validate_text(text)
    if error:
        return error, False

    task_line = f"\nThe writing task / question was: {task.strip()}\nAlso judge how well the text answers it.\n" if task and task.strip() else ""
    system = (
        "You are an experienced academic writing lecturer. You assess student writing fairly using a rubric, "
        "give specific evidence from the text, and offer practical suggestions. Be encouraging but honest."
    )
    user = f"""Assess the following {genre} using this rubric. Score each criterion from 1 (poor) to 5 (excellent).
{task_line}
Criteria:
1. Content & Main Idea (clear thesis/purpose, relevant and sufficient support)
2. Organization & Structure (introduction, topic sentences, logical order, conclusion)
3. Coherence & Cohesion (transitions, flow, reference words)
4. Vocabulary & Academic Style (precision, formality, variety)
5. Grammar & Mechanics (accuracy, punctuation, spelling)

Reply in exactly this format:

### Score Summary
| Criterion | Score (1-5) | Short Comment |
|---|---|---|
(one row per criterion)

**Total: X / 25**  |  **Level: (Beginner / Developing / Proficient / Advanced)**

### Strengths
(2-3 bullet points with evidence from the text)

### Areas to Improve
(3-4 bullet points, each with a concrete example from the text and how to fix it)

### Next Steps
(three prioritized actions the writer should do in the next revision)

TEXT:
\"\"\"{text}\"\"\""""
    return llm(system, user, temperature=0.3, max_tokens=2000)


# --------------------------------------------------------------------------
# Tool 4: Paraphraser
# --------------------------------------------------------------------------
def paraphrase(text, tone, keep_terms):
    text, error = validate_text(text)
    if error:
        return error, False

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

### What Changed
(2-3 bullet points explaining the main techniques used, e.g. synonym use, changing voice, restructuring clauses)

TEXT:
\"\"\"{text}\"\"\""""
    result, ok = llm(system, user, temperature=0.6, max_tokens=1800, use_cache=False)
    note = (
        "\n\n---\n> ⚖️ **Academic integrity reminder:** Paraphrasing an idea from another author still requires "
        "a citation. Use this tool to improve your own wording, not to avoid plagiarism detection."
    )
    return (result + note if ok else result), ok


# --------------------------------------------------------------------------
# Tool 5: Writing planner
# --------------------------------------------------------------------------
def plan_writing(topic, essay_type, level):
    topic, error = validate_text(topic)
    if error:
        return error, False

    system = (
        "You are an academic writing mentor. You help students plan before they write. "
        "You never invent citations, statistics or research findings."
    )
    user = f"""A student wants to write a {essay_type} (level: {level}) about this topic or question:

\"\"\"{topic}\"\"\"

Reply in exactly this format:

### Thesis Statement Options
(three different, specific, arguable thesis statements, each in one sentence, in academic English)

### Suggested Outline
(Introduction, 3 body paragraphs, Conclusion. For each: a topic sentence and 2 bullet points of ideas to develop.)

### Useful Keywords
(8-10 academic words or collocations the student can use)

### Common Mistakes to Avoid
(3 short bullet points specific to this kind of writing)

Do not add citations or statistics. Remind the student to find and cite real sources."""
    return llm(system, user, temperature=0.5, max_tokens=1800)


# --------------------------------------------------------------------------
# Tool 6: Text insights (no AI, always available)
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
PASSIVE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b", re.I)


def count_syllables(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
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
    content_words = [w for w in lower if w not in STOPWORDS and len(w) > 3]
    freq = {}
    for w in content_words:
        freq[w] = freq.get(w, 0) + 1
    repeated = sorted(((w, c) for w, c in freq.items() if c >= 3), key=lambda x: -x[1])[:8]
    low_text = text.lower()
    transitions = [t for t in TRANSITIONS if re.search(rf"\b{re.escape(t)}\b", low_text)]
    return {
        "words": n_words,
        "sentences": len(sentences),
        "avg_sentence": n_words / n_sent,
        "avg_word": sum(len(w) for w in words) / max(n_words, 1),
        "variety": len(set(lower)) / max(n_words, 1),
        "flesch": flesch,
        "long_sentences": [s for s in sentences if len(s.split()) > 30],
        "repeated": repeated,
        "transitions": transitions,
        "passive": [m.group(0) for m in PASSIVE.finditer(text)],
    }


def flesch_label(score):
    if score >= 70:
        return "Easy"
    if score >= 50:
        return "Fairly difficult"
    if score >= 30:
        return "Difficult (typical for academic writing)"
    return "Very difficult"


# --------------------------------------------------------------------------
# Tool 7: Phrasebank (no AI, always available)
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
    "Hedging (being careful with claims)": [
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
    "Evaluating sources and ideas": [
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
GENRES = ["academic paragraph", "academic essay", "abstract", "argumentative essay", "report"]
TONES = ["Formal academic", "Neutral academic", "Concise and direct"]
ESSAY_TYPES = ["argumentative essay", "opinion essay", "cause-and-effect essay", "compare-and-contrast essay",
               "problem-solution essay", "research paper introduction"]
LEVELS = ["beginner", "intermediate", "advanced"]
FEATURES = [
    "Grammar & Spelling Checker", "Academic Style Checker", "Rubric-Based Feedback",
    "Academic Paraphraser", "Writing Planner", "Text Insights", "Academic Phrasebank", "More than one feature",
]
RATING = [1, 2, 3, 4, 5]


def set_text(key, value):
    st.session_state[key] = value


def text_box(key, label="Your text", height=220, sample=None, placeholder="Paste your text here..."):
    if key not in st.session_state:
        st.session_state[key] = ""
    text = st.text_area(label, key=key, height=height, placeholder=placeholder)
    n = len(text.split())
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.caption(f"{n} / {MAX_WORDS} words")
    if sample is not None:
        col2.button("Load example", key=f"ex_{key}", on_click=set_text, args=(key, sample))
    col3.button("Clear", key=f"clr_{key}", on_click=set_text, args=(key, ""))
    return text


def run_tool(result_key, feature, spinner, fn, *args, words=0):
    """Run an AI tool, keep the result in session state, and log anonymous usage."""
    last = st.session_state.get("last_call", 0)
    if time.time() - last < COOLDOWN_SECONDS:
        st.warning("⏳ Please wait a few seconds before the next request.")
        return
    st.session_state["last_call"] = time.time()
    with st.spinner(spinner):
        result, ok = fn(*args)
    st.session_state[result_key] = result
    log_usage(feature, words, "ok" if ok else "error")


def show_result(result_key, filename):
    result = st.session_state.get(result_key)
    if not result:
        return
    with st.container(border=True):
        st.markdown(result)
    st.download_button("⬇️ Download result (.txt)", result, file_name=filename, key=f"dl_{result_key}")


def word_count(text):
    return len((text or "").split())


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write("**Wello** helps you improve your academic writing in English with open-source AI.")
    st.subheader("How to use")
    st.markdown(
        "1. Choose a tool from the tabs.\n"
        "2. Paste your text (or load the example).\n"
        "3. Click the button and read the feedback.\n"
        "4. Revise your text, then check it again."
    )
    st.subheader("Good to know")
    st.markdown(
        f"- Maximum **{MAX_WORDS} words** per request.\n"
        "- The AI can make mistakes. Always double-check.\n"
        "- If you see a *usage limit* message, wait a minute and try again.\n"
        "- **Text Insights** and **Phrasebank** work without AI."
    )
    st.subheader("Privacy")
    st.caption(
        "Your text is sent to a third-party AI service (Groq) only to generate feedback. "
        "This app does not store your text. Anonymous usage statistics (tool name and word count) "
        "are recorded for research."
    )
    st.caption(f"Model: `{MODEL_ID}`")


# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  {logo_html(64)}
  <div>
    <h1>Wello</h1>
    <p>Your free AI assistant for academic writing in English: check, improve, plan and learn.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if not GROQ_API_KEY:
    st.warning("The AI key is not set yet. Tools that use AI will not work until GROQ_API_KEY is added in Secrets. "
               "Text Insights and Phrasebank still work.")

tabs = st.tabs([
    "✅ Grammar", "🎓 Academic Style", "📝 Rubric Feedback", "🔁 Paraphraser",
    "🗺️ Planner", "📊 Insights", "📚 Phrasebank", "💬 Feedback",
])

# ---- 1. Grammar --------------------------------------------------------
with tabs[0]:
    st.markdown("Find grammar and spelling errors and learn **why** they are wrong.")
    t1 = text_box("t1", sample=SAMPLE)
    if st.button("Check Grammar", type="primary", key="b1"):
        run_tool("r1", "grammar", "Checking your text...", check_grammar, t1, words=word_count(t1))
    show_result("r1", "grammar_result.txt")

# ---- 2. Style ----------------------------------------------------------
with tabs[1]:
    st.markdown("Detect informal or vague language and get more academic alternatives.")
    t2 = text_box("t2", sample=SAMPLE)
    if st.button("Check Academic Style", type="primary", key="b2"):
        run_tool("r2", "style", "Analyzing your style...", check_style, t2, words=word_count(t2))
    show_result("r2", "style_result.txt")

# ---- 3. Rubric ---------------------------------------------------------
with tabs[2]:
    st.markdown("Get a score and detailed feedback on content, organization, coherence, vocabulary, and grammar.")
    c1, c2 = st.columns([1, 2])
    genre = c1.selectbox("Type of writing", GENRES, key="g3")
    task = c2.text_input("Writing task or question (optional)", key="task3",
                         placeholder="e.g. Discuss the advantages and disadvantages of online learning.")
    t3 = text_box("t3", height=280, placeholder="Paste your paragraph or essay here...")
    if st.button("Get Feedback", type="primary", key="b3"):
        run_tool("r3", "rubric", "Assessing your writing...", rubric_feedback, t3, genre, task, words=word_count(t3))
    show_result("r3", "rubric_feedback.txt")

# ---- 4. Paraphraser ----------------------------------------------------
with tabs[3]:
    st.markdown("Rewrite your own sentences in a more academic way. Three versions are provided.")
    t4 = text_box("t4", height=170, placeholder="Paste the sentence(s) to paraphrase...")
    c1, c2 = st.columns(2)
    tone = c1.selectbox("Tone", TONES, key="tone4")
    terms = c2.text_input("Key terms to keep (optional)", key="terms4", placeholder="e.g. motivation, second language")
    if st.button("Paraphrase", type="primary", key="b4"):
        run_tool("r4", "paraphrase", "Paraphrasing...", paraphrase, t4, tone, terms, words=word_count(t4))
    show_result("r4", "paraphrase_result.txt")

# ---- 5. Planner --------------------------------------------------------
with tabs[4]:
    st.markdown("Not sure how to start? Enter your topic and get thesis statements and an outline.")
    topic = st.text_area("Your topic or essay question", key="t5", height=110,
                         placeholder="e.g. The impact of social media on students' academic performance")
    c1, c2 = st.columns(2)
    essay_type = c1.selectbox("Type of writing", ESSAY_TYPES, key="et5")
    level = c2.selectbox("Your level", LEVELS, index=1, key="lv5")
    if st.button("Create Plan", type="primary", key="b5"):
        run_tool("r5", "planner", "Planning your essay...", plan_writing, topic, essay_type, level,
                 words=word_count(topic))
    show_result("r5", "writing_plan.txt")

# ---- 6. Insights -------------------------------------------------------
with tabs[5]:
    st.markdown("See statistics and readability of your text. This tool works **without AI** and is always available.")
    t6 = text_box("t6", sample=SAMPLE)
    if st.button("Analyze Text", type="primary", key="b6"):
        clean, err = validate_text(t6)
        if err:
            st.warning(err)
        else:
            st.session_state["r6"] = text_insights(clean)
            log_usage("insights", word_count(clean), "ok")
    info = st.session_state.get("r6")
    if info:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Words", info["words"])
        m2.metric("Sentences", info["sentences"])
        m3.metric("Avg. sentence length", f"{info['avg_sentence']:.1f}")
        m4.metric("Vocabulary variety", f"{info['variety']:.0%}")
        m5, m6, m7 = st.columns(3)
        m5.metric("Reading ease (Flesch)", f"{info['flesch']:.0f}")
        m6.metric("Avg. word length", f"{info['avg_word']:.1f}")
        m7.metric("Possible passive verbs", len(info["passive"]))
        st.caption(f"Reading level: **{flesch_label(info['flesch'])}**. Academic texts are often between 20 and 50.")

        with st.container(border=True):
            st.markdown("**Transition words used:** " + (", ".join(info["transitions"]) if info["transitions"]
                        else "none found. Try adding words such as *however*, *moreover*, or *therefore*."))
            st.markdown("**Frequently repeated words:** " + (", ".join(f"{w} ({c}×)" for w, c in info["repeated"])
                        if info["repeated"] else "no strong repetition found."))
            if info["long_sentences"]:
                st.markdown(f"**Long sentences (over 30 words): {len(info['long_sentences'])}.** Consider splitting them:")
                for s in info["long_sentences"][:5]:
                    st.markdown(f"> {s}")
            else:
                st.markdown("**Long sentences:** none over 30 words. Good!")
            if info["passive"]:
                st.markdown("**Possible passive structures:** " + "; ".join(f"*{p}*" for p in info["passive"][:8]))
        st.caption("These numbers are estimates based on simple rules, so use them as a guide, not as a grade.")

# ---- 7. Phrasebank -----------------------------------------------------
with tabs[6]:
    st.markdown("Ready-to-use academic phrases. Choose a category or search. Replace the dots (...) with your own ideas. "
                "This tool works **without AI**.")
    c1, c2 = st.columns([1, 1])
    category = c1.selectbox("Category", ["All categories"] + list(PHRASEBANK.keys()), key="pb_cat")
    query = c2.text_input("Search a word or phrase", key="pb_q", placeholder="e.g. however")
    shown = 0
    for cat, phrases in PHRASEBANK.items():
        if category != "All categories" and cat != category:
            continue
        matching = [p for p in phrases if query.strip().lower() in p.lower()]
        if not matching:
            continue
        st.markdown(f"##### {cat}")
        for p in matching:
            st.markdown(f'<div class="phrase">{p}</div>', unsafe_allow_html=True)
            shown += 1
    if shown == 0:
        st.info("No phrases found. Try another word or category.")
    st.markdown('<div class="tip">💡 Tip: copy a phrase, then complete it with your own ideas. '
                'Do not copy whole sentences from other authors.</div>', unsafe_allow_html=True)

# ---- 8. Feedback -------------------------------------------------------
with tabs[7]:
    st.markdown("Help us improve Wello. Your answers are used for research purposes.")
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name (optional)")
        c1, c2 = st.columns(2)
        role = c1.selectbox("You are a...", ["Student", "Lecturer / Teacher", "Other"])
        feature = c2.selectbox("Which feature did you use?", FEATURES)
        ease = st.radio("Ease of use (1 = very hard, 5 = very easy)", RATING, index=None, horizontal=True)
        useful = st.radio("Usefulness (1 = not useful, 5 = very useful)", RATING, index=None, horizontal=True)
        accuracy = st.radio("Accuracy of feedback (1 = poor, 5 = excellent)", RATING, index=None, horizontal=True)
        design = st.radio("Appearance / design (1 = poor, 5 = excellent)", RATING, index=None, horizontal=True)
        comment = st.text_area("Comments or suggestions", height=100)
        submitted = st.form_submit_button("Submit Feedback", type="primary")

    if submitted:
        if None in (ease, useful, accuracy, design):
            st.warning("⚠️ Please rate all four aspects before submitting.")
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
            st.success("✅ Thank you! Your feedback has been saved.")

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
                st.error("❌ Wrong password, or ADMIN_PASSWORD is not set.")

st.divider()
st.caption("Wello · Open-source AI · Developed for educational research")
