import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random

API_KEY = st.secrets["GROQ_API_KEY"]
headers = {"Authorization": f"Bearer {API_KEY}"}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def generate_prompt(text_chunk):
    return f"""
You are an educational assistant helping teachers generate multiple choice questions from a passage.

Given the following passage or notes, generate exactly 15 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 15 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage. If the passage is too short to reasonably support 15 distinct questions, generate as many high-quality questions as possible (minimum of 5), ensuring they reflect varying difficulty.

**Requirements**:
- 5 easy (≥85%), 5 medium (60–84%), 5 hard (<60%)

**Each question must include the following fields:**

- "question": A clear, concise, and unambiguous question directly related to the passage that aligns with key learning objectives. The question should be designed to test understanding of material covered in the passage and should be made so it could show up on an educational assessment testing material from this passage. The question should be cognitively appropriate for the specified difficulty level, encouraging critical thinking, application, analysis, or synthesis rather than rote recall if not at the easiest difficulty level. Avoid overly complex wording or ambiguity to ensure students understand exactly what is being asked. Furthermore, make sure that the question has all the context in itself and does not reference specific figures or pages in the passage, as the question is designed for the user to do independently without the passage. If the question involves math equations, use latex when generating the question.
- "options": A list of 4 plausible answer choices labeled "A", "B", "C", and "D"(with one of them being the correct answer). If the question is of medium or hard difficulty, please come up with wrong answers that are ones that a user who does not know the concept well or makes an error would select. Please make sure that only one answer choice is correct by solving the problem and checking all of the answer choices carefully and thoroughly. It should not be ambigiuous which as to whether an answer choice is correct or not. If the answer involves math, make sure that the answer is formatted in latex to improve readability.
- "correct_answer": The letter ("A", "B", "C", or "D") corresponding to the correct option.
- "explanation": A deep, pedagogically useful explanation that **teaches the concept** behind the correct answer and analyzes the flaws in the others. The explanation must:
    1. Start by stating the correct letter and full answer.
    2. Teach **why** that answer is correct using **conceptual reasoning** — including how the mechanism works, or why the property matters — not just restating facts.
       - For example, if the correct answer is "correlation between features degrades performance," then explain **why correlated features reduce tree diversity in Random Forests**, which is the core reason performance drops.
       - Use step-by-step reasoning, examples, or analogies when helpful.
    3. For each incorrect answer, state its letter and text, and **explain why it's wrong**, including what misconception a student might have that could lead them to choose it.
    4. The tone should be that of a **tutor or explainer**, helping a confused student understand both the correct idea and the traps in the wrong ones.

Avoid vague phrases like “According to the passage.” Don’t just repeat the answer. Your goal is to help the student learn the concept by explaining it clearly and thoroughly.
- "estimated_correct_pct": A numeric estimate of the percentage of students expected to answer correctly (consistent with the difficulty category). Make it based on factors such as complexity, inference required, or detail recall.
- "reasoning": A brief rationale explaining why the question fits its percentage correct assignment, considering factors such as complexity, inference required, or detail recall.

Return a valid JSON list of up to 15 questions. If there is insufficient content, generate as many high-quality questions as possible (minimum 5).

Passage:
{text_chunk}
"""

def call_groq_api(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4500
    }
    response = requests.post(GROQ_URL, headers=headers, json=data)
    if response.status_code != 200:
        return None, response.text
    return response.json()["choices"][0]["message"]["content"], None

def clean_response_text(text):
    match = re.search(r"```(?:json)?\s*(.*?)```", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def parse_question_json(text):
    try:
        return json.loads(clean_response_text(text))
    except Exception:
        return []

def assign_difficulty_label(estimated_pct):
    try:
        pct = int(estimated_pct)
    except:
        return None
    if pct < 30: return 8
    elif pct < 40: return 7
    elif pct < 50: return 6
    elif pct < 65: return 5
    elif pct < 75: return 4
    elif pct < 85: return 3
    elif pct < 90: return 2
    else: return 1

def group_by_difficulty(questions):
    groups = {i: [] for i in range(1, 9)}
    for q in questions:
        pct = q.get("estimated_correct_pct", 0)
        label = assign_difficulty_label(pct)
        if label:
            q["difficulty_label"] = label
            groups[label].append(q)
    return groups

def pick_question(diff, asked, all_qs):
    pool = all_qs.get(diff, [])
    return [(i, q) for i, q in enumerate(pool) if (diff, i) not in asked]

def get_next_question(target_diff, asked, all_qs, going_up=True):
    candidates = pick_question(target_diff, asked, all_qs)
    if candidates:
        return target_diff, *random.choice(candidates)
    range_fn = (
        lambda: range(target_diff + 1, 9) if going_up else range(target_diff - 1, 0, -1)
    )
    for d in range_fn():
        candidates = pick_question(d, asked, all_qs)
        if candidates:
            return d, *random.choice(candidates)
    for d in range(1, 9):
        candidates = pick_question(d, asked, all_qs)
        if candidates:
            return d, *random.choice(candidates)
    return None, None, None

def accuracy_on_levels(answers, levels):
    filtered = [c for d, c in answers if d in levels]
    return sum(filtered) / len(filtered) if filtered else 0

def compute_mastery_score(answers):
    mastery_structure = {
        (1, 2): 0.15,
        (3, 4): 0.25,
        (5, 6): 0.25,
        (7, 8): 0.30
    }
    total_score = 0.0
    total_weight = 0.0
    for levels, weight in mastery_structure.items():
        acc = accuracy_on_levels(answers, levels)
        if acc is not None:
            adjusted = max((acc - 0.25) / 0.75, 0)
            total_score += weight * adjusted
            total_weight += weight
    if total_weight == 0:
        return 0
    return int(round((total_score / total_weight) * 100))

def render_latex_or_markdown(text):
    # If the whole text is a single LaTeX expression
    if re.fullmatch(r"\$.*\$", text.strip()):
        st.latex(text.strip('$'))  # Strip outer $ only
    else:
        st.markdown(text, unsafe_allow_html=True)  # Supports embedded LaTeX
