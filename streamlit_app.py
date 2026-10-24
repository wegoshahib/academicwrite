"""
AcademiWrite - an open-source AI assistant for academic writing in English.
Built with Streamlit. Runs for free on Streamlit Community Cloud.

Features
1. Grammar & Spelling Checker (with simple explanations)
2. Academic Style Checker
3. Rubric-Based Feedback
4. Academic Paraphraser
5. User Feedback Form (for research data)
"""

import csv
import os
import re
from datetime import datetime, timezone

import requests
import streamlit as st

st.set_page_config(page_title="AcademiWrite", page_icon="✍️", layout="centered")


# --------------------------------------------------------------------------
# Configuration (set these in Streamlit "Settings > Secrets")
# --------------------------------------------------------------------------
def get_secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:  # noqa: BLE001
        return os.environ.get(name, default)


MODEL_ID = get_secret("MODEL_ID", "openai/gpt-oss-120b")
GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_TOKEN = get_secret("HF_TOKEN", "")  # optional, only for saving feedback to a HF dataset
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD", "")
FEEDBACK_DATASET = get_secret("FEEDBACK_DATASET", "")  # e.g. "username/academiwrite-feedback"
LANGUAGETOOL_URL = "https://api.languagetool.org/v2/check"
MAX_WORDS = 1200
FEEDBACK_FILE = "feedback.csv"
FEEDBACK_FIELDS = [
    "timestamp", "name", "role", "feature_used",
    "ease_of_use", "usefulness", "accuracy", "design", "comment",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def llm(system_prompt, user_prompt, temperature=0.3, max_tokens=1800):
    """Call the open-source language model. Returns text or a friendly error."""
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_tokens + 1500,
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        detail = str(exc) + ("\n" + resp.text if resp is not None else "")
        if GROQ_API_KEY:
            detail = detail.replace(GROQ_API_KEY, "***")
        hints = {
            400: "The request was rejected. The model name in MODEL_ID may be wrong.",
            401: "The API key is missing or invalid. Check GROQ_API_KEY in Secrets.",
            403: "Access denied. Check your Groq account and API key.",
            404: "The model is not available. Set MODEL_ID in Secrets to another model.",
            413: "The text is too long for the free limit. Please use a shorter text.",
            429: "The free limit is reached. Please wait a minute and try again.",
        }
        return (
            "⚠️ The AI service is busy or unavailable right now. "
            "Please wait a moment and try again.\n\n"
            f"(Technical detail: {type(exc).__name__}, status {status})\n\n"
            f"{hints.get(status, '')}\n\n"
            f"```\n{detail[:400]}\n```"
        )


def validate_text(text):
    """Return (clean_text, error_message)."""
    text = (text or "").strip()
    if not text:
        return None, "⚠️ Please paste or type your text first."
    if len(text.split()) > MAX_WORDS:
        return None, f"⚠️ Your text is too long. Please keep it under {MAX_WORDS} words."
    return text, None


# --------------------------------------------------------------------------
# Feature 1: Grammar & Spelling Checker
# --------------------------------------------------------------------------
def languagetool_check(text):
    try:
        r = requests.post(
            LANGUAGETOOL_URL,
            data={"text": text, "language": "en-US"},
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("matches", [])
    except Exception:  # noqa: BLE001
        return None


def check_grammar(text):
    text, error = validate_text(text)
    if error:
        return error

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
# Feature 2: Academic Style Checker
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
        return error

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
    ai_part = llm(system, user, temperature=0.3, max_tokens=2200)

    header = "### 🔎 Automatic Scan Results\n" + (scan if scan else "No obvious informal expressions found.") + "\n\n---\n\n"
    return header + ai_part


# --------------------------------------------------------------------------
# Feature 3: Rubric-Based Feedback
# --------------------------------------------------------------------------
def rubric_feedback(text, genre):
    text, error = validate_text(text)
    if error:
        return error

    system = (
        "You are an experienced academic writing lecturer. You assess student writing fairly using a rubric, "
        "give specific evidence from the text, and offer practical suggestions. Be encouraging but honest."
    )
    user = f"""Assess the following {genre} using this rubric. Score each criterion from 1 (poor) to 5 (excellent).

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
# Feature 4: Academic Paraphraser
# --------------------------------------------------------------------------
def paraphrase(text, tone, keep_terms):
    text, error = validate_text(text)
    if error:
        return error

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
    result = llm(system, user, temperature=0.6, max_tokens=1800)
    note = (
        "\n\n---\n> ⚖️ **Academic integrity reminder:** Paraphrasing an idea from another author still requires "
        "a citation. Use this tool to improve your own wording, not to avoid plagiarism detection."
    )
    return result + note


# --------------------------------------------------------------------------
# Feature 5: User Feedback Form (research data)
# --------------------------------------------------------------------------
def push_to_dataset(row):
    """Optionally keep a permanent copy in a private HF dataset repo."""
    if not (FEEDBACK_DATASET and HF_TOKEN):
        return
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=HF_TOKEN)
        rows = []
        try:
            path = hf_hub_download(
                repo_id=FEEDBACK_DATASET, filename=FEEDBACK_FILE,
                repo_type="dataset", token=HF_TOKEN, force_download=True,
            )
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:  # noqa: BLE001
            rows = []
        rows.append(row)
        tmp = "feedback_upload.csv"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        api.upload_file(
            path_or_fileobj=tmp, path_in_repo=FEEDBACK_FILE,
            repo_id=FEEDBACK_DATASET, repo_type="dataset",
        )
    except Exception:  # noqa: BLE001
        pass


def save_feedback(row):
    new_file = not os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    push_to_dataset(row)


# --------------------------------------------------------------------------
# User Interface
# --------------------------------------------------------------------------
SAMPLE = (
    "Nowadays, many students dont know how to write a good essay. "
    "They get a lot of problems when they writes the introduction, and I think this is a big thing."
)
GENRES = ["academic paragraph", "academic essay", "abstract", "argumentative essay", "report"]
TONES = ["Formal academic", "Neutral academic", "Concise and direct"]
FEATURES = [
    "Grammar & Spelling Checker", "Academic Style Checker", "Rubric-Based Feedback",
    "Academic Paraphraser", "More than one feature",
]
RATING = [1, 2, 3, 4, 5]

st.title("✍️ AcademiWrite")
st.subheader("Your free AI assistant for academic writing in English")
st.caption("Paste your text, choose a tool, and get instant feedback. Built with open-source AI. "
           "💡 The AI can make mistakes, so always double-check important corrections.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Grammar", "2. Academic Style", "3. Rubric Feedback", "4. Paraphraser", "5. Give Feedback",
])

with tab1:
    st.write("Find grammar and spelling errors and learn **why** they are wrong.")
    text1 = st.text_area("Your text", value=SAMPLE, height=200, key="t1", placeholder="Paste your text here...")
    if st.button("Check Grammar", type="primary", key="b1"):
        with st.spinner("Checking your text..."):
            st.markdown(check_grammar(text1))

with tab2:
    st.write("Detect informal or vague language and get more academic alternatives.")
    text2 = st.text_area("Your text", value=SAMPLE, height=200, key="t2", placeholder="Paste your text here...")
    if st.button("Check Academic Style", type="primary", key="b2"):
        with st.spinner("Analyzing your style..."):
            st.markdown(check_style(text2))

with tab3:
    st.write("Get a score and detailed feedback on content, organization, coherence, vocabulary, and grammar.")
    genre = st.selectbox("Type of writing", GENRES, key="g3")
    text3 = st.text_area("Your text", height=260, key="t3", placeholder="Paste your paragraph or essay here...")
    if st.button("Get Feedback", type="primary", key="b3"):
        with st.spinner("Assessing your writing..."):
            st.markdown(rubric_feedback(text3, genre))

with tab4:
    st.write("Rewrite your sentences in a more academic way. Three versions are provided.")
    text4 = st.text_area("Your text", height=160, key="t4", placeholder="Paste the sentence(s) to paraphrase...")
    tone = st.selectbox("Tone", TONES, key="tone4")
    terms = st.text_input("Key terms to keep (optional, separate with commas)", key="terms4",
                          placeholder="e.g. second language acquisition, motivation")
    if st.button("Paraphrase", type="primary", key="b4"):
        with st.spinner("Paraphrasing..."):
            st.markdown(paraphrase(text4, tone, terms))

with tab5:
    st.write("Help us improve AcademiWrite. Your answers are used for research purposes.")
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name (optional)")
        role = st.selectbox("You are a...", ["Student", "Lecturer / Teacher", "Other"])
        feature = st.selectbox("Which feature did you use?", FEATURES)
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
            save_feedback({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "name": (name or "").strip() or "Anonymous",
                "role": role,
                "feature_used": feature,
                "ease_of_use": ease,
                "usefulness": useful,
                "accuracy": accuracy,
                "design": design,
                "comment": (comment or "").strip(),
            })
            st.success("✅ Thank you! Your feedback has been saved.")

    with st.expander("Admin: download collected data"):
        pw = st.text_input("Admin password", type="password", key="admin_pw")
        if pw:
            if ADMIN_PASSWORD and pw == ADMIN_PASSWORD:
                if os.path.exists(FEEDBACK_FILE):
                    with open(FEEDBACK_FILE, "rb") as f:
                        st.download_button("Download feedback.csv", f.read(),
                                           file_name="feedback.csv", mime="text/csv")
                else:
                    st.info("No feedback has been collected yet.")
            else:
                st.error("❌ Wrong password, or ADMIN_PASSWORD is not set.")

st.divider()
st.caption("AcademiWrite · Open-source AI · Developed for educational research")
