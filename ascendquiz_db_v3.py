import streamlit as st
import sqlite3
import json
import random
import re
import requests
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF

# ============== CONFIGURATION ==============

API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"

DB_PATH = "ascendquiz.db"

QUESTION_POOL_SIZE = 30
QUIZ_LENGTH = 20

# Number of questions per tier (easy, medium, medium-hard, hard) — always sums to 30
POOL_DISTRIBUTIONS = {
    "Easy":   (12, 10, 6,  2),
    "Medium": (8,  7,  8,  7),
    "Hard":   (2,  6,  10, 12),
}

# Starting tier for adaptive algorithm (1=easiest, 4=hardest)
STARTING_TIER = {
    "Easy":   1,
    "Medium": 2,
    "Hard":   3,
}

# Tier → (min_correct_pct, max_correct_pct)
TIERS = {
    1: (75, 100),
    2: (50, 74),
    3: (30, 49),
    4: (0,  29),
}
TIER_NAMES = {1: "Easy", 2: "Medium", 3: "Medium-Hard", 4: "Hard"}


# ============== DATABASE ==============

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Migrate quiz_sessions if it exists with the old schema
    c.execute("PRAGMA table_info(quiz_sessions)")
    columns = {row[1] for row in c.fetchall()}
    if columns and "correct_answers" not in columns:
        c.execute("DROP TABLE IF EXISTS quiz_sessions")

    c.execute('''CREATE TABLE IF NOT EXISTS quiz_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        pdf_name TEXT,
        correct_answers INTEGER,
        total_questions INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')

    # Drop legacy tables if they exist
    c.execute("DROP TABLE IF EXISTS responses")
    c.execute("DROP TABLE IF EXISTS topic_stats")

    conn.commit()
    conn.close()

def create_user(username):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError:
        conn.close()
        return None, "Username already exists"

def get_user(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def save_quiz_session(user_id, pdf_name, correct_answers, total_questions):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO quiz_sessions (user_id, pdf_name, correct_answers, total_questions) VALUES (?, ?, ?, ?)",
        (user_id, pdf_name, correct_answers, total_questions)
    )
    conn.commit()
    conn.close()

init_db()


# ============== PDF PROCESSING ==============

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def get_chunks_by_token(pages):
    full_text = "\n\n".join(pages)
    TOKEN_CHUNK_SIZE = 10000 * 4  # ~40,000 characters per chunk
    all_chunks = [full_text[i:i + TOKEN_CHUNK_SIZE] for i in range(0, len(full_text), TOKEN_CHUNK_SIZE)]
    if len(all_chunks) <= 2:
        return all_chunks
    return random.sample(all_chunks, 2)


# ============== GEMINI API ==============

def call_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 16000,
        }
    }
    url = f"{GEMINI_URL}?key={API_KEY}"
    response = requests.post(url, headers=headers, json=data, timeout=120)
    if response.status_code != 200:
        return None, response.text
    response_json = response.json()
    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"], None
    except (KeyError, IndexError) as e:
        return None, f"Failed to parse Gemini response: {str(e)}"

def clean_response_text(text: str) -> str:
    text = text.strip()
    for pattern in [r"```json\s*(.*?)```", r"```\s*(.*?)```", r"`{3,}\s*(.*?)`{3,}"]:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            break
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text

def repair_json(text: str) -> str:
    text = re.sub(r'```(?:json)?', '', text).replace('```', '').strip()
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        text = text[start:end + 1]
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = f"[{text[start:end + 1]}]"
    open_b = text.count('{')
    close_b = text.count('}')
    if close_b < open_b:
        last = text.rfind('}')
        if last != -1:
            text = text[:last + 1]
        if not text.endswith(']'):
            text += "]"
    text = re.sub(r'}\s*{', '}, {', text)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    if not text.startswith('['):
        text = '[' + text
    if not text.endswith(']'):
        text = text + ']'
    return text.strip()

def parse_question_json(text: str):
    cleaned = repair_json(clean_response_text(text))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            import json5
            return json5.loads(cleaned)
        except Exception:
            questions = []
            for q_text in re.findall(r'\{[^{}]*?"question"[^{}]*?\}', cleaned, re.DOTALL):
                try:
                    questions.append(json.loads(q_text))
                except Exception:
                    pass
            return questions


# ============== QUESTION GENERATION ==============

def generate_batch_prompt(text_chunk, tier, count):
    low, high = TIERS[tier]

    difficulty_instructions = {
        1: f"""These are EASY questions (estimated_correct_pct must be between {low} and {high}).
Require direct recall of facts, definitions, or formulas explicitly stated in the passage.
No calculation or inference needed beyond what is written.
Distractors should be small, plausible misconceptions — not absurdities.""",

        2: f"""These are MEDIUM questions (estimated_correct_pct must be between {low} and {high}).
Require 1–2 steps of reasoning or minor inference beyond direct recall.
Students must connect a concept to a similar context or combine two pieces of information.
Wrong answers should reflect adjacent concepts that partial understanding might confuse.""",

        3: f"""These are MEDIUM-HARD questions (estimated_correct_pct must be between {low} and {high}).
Require applying principles to new scenarios or integrating multiple pieces of information.
The passage does not explicitly solve this — students must adapt their knowledge.
Distractors: plausible errors from misapplied rules or partial understanding.""",

        4: f"""These are HARD questions (estimated_correct_pct must be between {low} and {high}).
Require deep analysis, synthesis, or prediction combining multiple concepts.
Students must infer relationships, compare methods, or predict outcomes not directly stated.
Wrong answers should exploit subtle distinctions and seem correct to those with partial understanding.""",
    }[tier]

    return f"""You are a teacher designing a multiple-choice test to assess understanding of a passage.

{difficulty_instructions}

Generate exactly {count} multiple-choice questions based on the passage below.

**CRITICAL — NO TEXT REFERENCES:**
- Questions must be COMPLETELY SELF-CONTAINED
- Never use phrases like "according to the passage," "the text states," "as mentioned," etc.
- Never reference specific figures, tables, pages, or sections from the passage
- Students should answer based on conceptual understanding, not text location
- Do not ask about ISBN, copyright, or distribution information

**CRITICAL — TEST TRUE MASTERY, NOT TEST-TAKING SKILLS:**
- Wrong answers must represent genuine domain misconceptions, not obvious nonsense
- All 4 options must be similar in length, specificity, and technical complexity
- The correct answer must not be guessable from wording or option patterns
- Students who memorized without understanding should NOT reliably get these correct

**REQUIRED JSON FIELDS per question:**
- "question": Self-contained, unambiguous question testing a concept from the passage
- "options": Array of exactly 4 strings in format ["A. ...", "B. ...", "C. ...", "D. ..."]
    * All options similar in length, specificity, grammatical structure, and technicality
    * Wrong answers represent genuine misconceptions from the domain
- "correct_answer": Letter ("A", "B", "C", or "D")
- "explanation_correct": A concise explanation of WHY the correct answer is right.
    Explain the underlying mechanism, principle, or concept — not just that it is correct.
    Write directly to the student in plain, clear language.
    NEVER include meta-commentary or internal reasoning such as "I need to", "Let me",
    "We should consider", "Looking at the options", or any similar phrases.
- "explanation_wrong": An object mapping each wrong answer letter to a brief explanation
    of why that option is wrong and what misconception it represents.
    Include only the 3 wrong answer letters (not the correct one). Example format:
    {{"B": "why B is wrong...", "C": "why C is wrong...", "D": "why D is wrong..."}}
    Each value must be written directly to the student with no meta-commentary.
- "estimated_correct_pct": Integer between {low} and {high} — MUST stay in this range

All math expressions must use valid LaTeX format ($...$ for inline, $$...$$ for display math).
Before finalizing each question, verify the correct answer is clearly correct and each wrong answer is clearly incorrect based on the passage content.

Return ONLY a valid JSON array of exactly {count} questions.
Output must begin with `[` and end with `]` — no text, commentary, or markdown outside the JSON.

Passage:
{text_chunk}
"""

def generate_question_pool(text_chunk, difficulty_mode):
    """
    Generate exactly QUESTION_POOL_SIZE questions via 4 concurrent API calls,
    one per difficulty tier. Returns dict: {tier: [questions]}.
    """
    easy_n, med_n, mh_n, hard_n = POOL_DISTRIBUTIONS[difficulty_mode]
    tier_counts = {1: easy_n, 2: med_n, 3: mh_n, 4: hard_n}

    def fetch_tier(tier, count):
        questions = []
        for _ in range(3):  # up to 2 retries per tier
            needed = count - len(questions)
            if needed <= 0:
                break
            prompt = generate_batch_prompt(text_chunk, tier, needed)
            raw, error = call_gemini_api(prompt)
            if error or not raw:
                continue
            parsed = parse_question_json(raw)
            for q in parsed:
                if isinstance(q, dict):
                    q["difficulty_tier"] = tier
                    questions.append(q)
            if len(questions) >= count:
                break
        return questions[:count]

    pool_by_tier = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {tier: executor.submit(fetch_tier, tier, cnt)
                   for tier, cnt in tier_counts.items()}
        for tier, future in futures.items():
            pool_by_tier[tier] = future.result()

    return pool_by_tier


# ============== ADAPTIVE ENGINE ==============

def pick_question(tier, asked, pool_by_tier):
    """Return list of (index, question) pairs not yet asked at this tier."""
    pool = pool_by_tier.get(tier, [])
    return [(i, q) for i, q in enumerate(pool) if (tier, i) not in asked]

def find_next_tier(current_tier, going_up, asked, pool_by_tier):
    """Find next tier with available questions in the desired direction."""
    target = current_tier + 1 if going_up else current_tier - 1
    target = max(1, min(4, target))

    if pick_question(target, asked, pool_by_tier):
        return target

    # Search further in the desired direction
    search = range(target + 1, 5) if going_up else range(target - 1, 0, -1)
    for t in search:
        if pick_question(t, asked, pool_by_tier):
            return t

    # Fall back to opposite direction from current
    fallback = range(current_tier - 1, 0, -1) if going_up else range(current_tier + 1, 5)
    for t in fallback:
        if pick_question(t, asked, pool_by_tier):
            return t

    return current_tier

def get_next_question(current_tier, asked, pool_by_tier):
    """Pick a random unanswered question at the current tier."""
    available = pick_question(current_tier, asked, pool_by_tier)
    if not available:
        return current_tier, None, None
    idx, q = random.choice(available)
    return current_tier, idx, q


# ============== PAGE CONFIG ==============

st.set_page_config(page_title="AscendQuiz", page_icon="📚", layout="centered")


# ============== UI ==============

def render_login_page():
    st.title("AscendQuiz")
    st.markdown("#### AI-powered adaptive quizzes from your study materials")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Returning User")
        login_user = st.text_input("Username", key="login_username", placeholder="Enter your username")
        if st.button("Login", key="login_btn", use_container_width=True):
            if login_user:
                user = get_user(login_user)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("User not found. Please sign up first.")
            else:
                st.warning("Please enter a username.")

    with col2:
        st.markdown("#### New User")
        new_user = st.text_input("Choose a Username", key="signup_username", placeholder="Pick a username")
        if st.button("Create Account", key="signup_btn", use_container_width=True):
            if new_user:
                if len(new_user) < 3:
                    st.error("Username must be at least 3 characters.")
                else:
                    user_id, error = create_user(new_user)
                    if user_id:
                        st.session_state.user = {"id": user_id, "username": new_user}
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error(f"{error}")
            else:
                st.warning("Please choose a username.")


def render_home():
    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.title("AscendQuiz")
        st.caption(f"Logged in as **{st.session_state.user['username']}**")
    with col_logout:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### Upload Your Study Material")
    st.caption("Upload a PDF and AscendQuiz will generate an adaptive 20-question quiz tailored to your performance.")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown("**Difficulty Mode**")
    difficulty_mode = st.radio(
        "Difficulty",
        ["Easy", "Medium", "Hard"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.caption({
        "Easy": "Starts with easier questions. Good for initial review.",
        "Medium": "Balanced distribution. Recommended for most use cases.",
        "Hard": "Starts with harder questions. For confident learners.",
    }[difficulty_mode])

    st.markdown("")

    if uploaded_pdf:
        if st.button("Generate Quiz", use_container_width=True, type="primary"):
            with st.spinner("Extracting text from PDF..."):
                try:
                    pages = extract_text_from_pdf(uploaded_pdf)
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")
                    st.stop()

            st.session_state.pdf_pages = pages
            st.session_state.pdf_name = uploaded_pdf.name
            st.session_state.difficulty_mode = difficulty_mode
            st.session_state.pool_by_tier = None  # triggers generation in render_quiz
            st.session_state.quiz_active = True
            st.session_state.quiz_done = False
            st.rerun()
    else:
        st.button("Generate Quiz", use_container_width=True, type="primary", disabled=True)


def render_quiz():
    state = st.session_state.get("quiz_state")

    # ---- Generation phase ----
    if st.session_state.get("pool_by_tier") is None:
        st.title("Generating Your Quiz")

        pages = st.session_state.pdf_pages
        difficulty_mode = st.session_state.difficulty_mode
        chunks = get_chunks_by_token(pages)
        text_chunk = chunks[0] if chunks else ""

        if not text_chunk.strip():
            st.error("Could not extract text from the PDF. Please try a different file.")
            if st.button("Back to Home"):
                _clear_quiz_state()
                st.rerun()
            st.stop()

        with st.spinner("Building your question pool — this takes about 20–30 seconds..."):
            try:
                pool = generate_question_pool(text_chunk, difficulty_mode)
            except Exception as e:
                st.error(f"Error generating questions: {e}")
                if st.button("Back to Home"):
                    _clear_quiz_state()
                    st.rerun()
                st.stop()

        # Validate: need at least QUIZ_LENGTH questions total
        total_generated = sum(len(v) for v in pool.values())
        if total_generated < QUIZ_LENGTH:
            st.error(
                f"Only {total_generated} questions were generated (need {QUIZ_LENGTH}). "
                "Please try a different PDF or difficulty mode."
            )
            if st.button("Back to Home"):
                _clear_quiz_state()
                st.rerun()
            st.stop()

        st.session_state.pool_by_tier = pool
        st.session_state.quiz_state = {
            "current_tier": STARTING_TIER[difficulty_mode],
            "asked": set(),
            "answers": [],          # list of (tier, was_correct)
            "question_number": 0,   # questions answered so far
            "current_q": None,
            "current_q_idx": None,
            "show_explanation": False,
            "last_correct": None,
        }
        st.rerun()
        return

    # ---- Active quiz ----
    pool_by_tier = st.session_state.pool_by_tier
    state = st.session_state.quiz_state

    # Load next question if needed
    if state["current_q"] is None and not state["show_explanation"]:
        tier, idx, q = get_next_question(state["current_tier"], state["asked"], pool_by_tier)
        if q is None:
            # No questions available at current tier — try to find any remaining
            found = False
            for t in range(1, 5):
                tier, idx, q = get_next_question(t, state["asked"], pool_by_tier)
                if q is not None:
                    state["current_tier"] = t
                    found = True
                    break
            if not found:
                # Pool exhausted before QUIZ_LENGTH — shouldn't happen with good generation
                _finish_quiz()
                return
        state["current_q"] = q
        state["current_q_idx"] = idx
        state["current_tier"] = tier

    q = state["current_q"]
    if q is None:
        _finish_quiz()
        return

    num_answered = state["question_number"]
    num_correct = sum(1 for _, c in state["answers"] if c)
    tier = state["current_tier"]

    # Progress header
    st.markdown(
        f"**Question {num_answered + 1} of {QUIZ_LENGTH}** &nbsp;|&nbsp; "
        f"{num_correct} correct &nbsp;|&nbsp; Difficulty: {TIER_NAMES[tier]}"
    )
    st.progress((num_answered) / QUIZ_LENGTH)
    st.markdown("---")

    # Question text
    st.markdown(f"### {q['question']}")

    if not state["show_explanation"]:
        def strip_label(text):
            return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()

        option_labels = ["A", "B", "C", "D"]
        cleaned = [strip_label(opt) for opt in q["options"]]
        rendered = []
        for label, text in zip(option_labels, cleaned):
            rendered.append(f"{label}. $${text}$$" if ("$" in text or "\\" in text) else f"{label}. {text}")

        selected = st.radio("Choose your answer:", options=rendered, key=f"q_{num_answered}", index=None)

        if st.button("Submit Answer", use_container_width=True):
            if selected is None:
                st.warning("Please select an answer.")
            else:
                selected_letter = selected.split(".")[0].strip().upper()
                correct_letter = q["correct_answer"].strip().upper()
                was_correct = (selected_letter == correct_letter)

                state["asked"].add((tier, state["current_q_idx"]))
                state["answers"].append((tier, was_correct))
                state["question_number"] += 1
                state["last_correct"] = was_correct
                state["show_explanation"] = True
                st.rerun()
    else:
        # Feedback
        correct_letter = q["correct_answer"].strip().upper()

        if state["last_correct"]:
            st.success("Correct!")
        else:
            st.error(f"Incorrect. The correct answer was **{correct_letter}**.")
        explanation_correct = q.get("explanation_correct") or q.get("explanation", "No explanation available.")
        explanation_wrong = q.get("explanation_wrong", {})

        st.markdown("**Why this is correct:**")
        st.info(explanation_correct)

        wrong_items = {k.upper(): v for k, v in explanation_wrong.items() if k.upper() != correct_letter}
        if wrong_items:
            st.markdown("**Why the other options are wrong:**")
            for letter in ["A", "B", "C", "D"]:
                if letter == correct_letter or letter not in wrong_items:
                    continue
                st.markdown(f"**{letter}.** {wrong_items[letter]}")

        is_last = (state["question_number"] >= QUIZ_LENGTH)
        btn_label = "See Results" if is_last else "Next Question"

        if st.button(btn_label, use_container_width=True):
            if is_last:
                _finish_quiz()
                return

            # Adjust difficulty for next question
            state["current_tier"] = find_next_tier(
                tier,
                going_up=state["last_correct"],
                asked=state["asked"],
                pool_by_tier=pool_by_tier,
            )
            state["current_q"] = None
            state["current_q_idx"] = None
            state["show_explanation"] = False
            state["last_correct"] = None
            st.rerun()


def _finish_quiz():
    """Save result and transition to results screen."""
    state = st.session_state.get("quiz_state", {})
    answers = state.get("answers", [])
    correct = sum(1 for _, c in answers if c)
    total = len(answers)

    save_quiz_session(
        st.session_state.user["id"],
        st.session_state.get("pdf_name", "Unknown"),
        correct,
        total,
    )

    st.session_state.quiz_result = {"correct": correct, "total": total}
    st.session_state.quiz_active = False
    st.session_state.quiz_done = True
    st.rerun()


def render_results():
    result = st.session_state.get("quiz_result", {})
    correct = result.get("correct", 0)
    total = result.get("total", QUIZ_LENGTH)
    pct = round(correct / total * 100) if total > 0 else 0

    st.title("Quiz Complete!")
    st.markdown("---")

    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 30px 0;">
                <div style="font-size: 4em; font-weight: bold; color: #1f77b4;">{correct}/{total}</div>
                <div style="font-size: 1.5em; color: #666; margin-top: 8px;">{pct}% correct</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        if st.button("Back to Home", use_container_width=True, type="primary"):
            _clear_quiz_state()
            st.rerun()
    with col_r:
        pdf_name = st.session_state.get("pdf_name")
        if pdf_name and st.button("Retry Same PDF", use_container_width=True):
            # Keep pdf_pages and difficulty_mode, reset pool and quiz state
            st.session_state.pool_by_tier = None
            st.session_state.quiz_active = True
            st.session_state.quiz_done = False
            st.session_state.quiz_state = None
            st.session_state.quiz_result = None
            st.rerun()

    pool = st.session_state.get("pool_by_tier")
    if pool:
        all_questions = [q for tier_qs in pool.values() for q in tier_qs]
        st.download_button(
            label="Download Question Pool (JSON)",
            data=json.dumps(all_questions, indent=2),
            file_name="ascendquiz_questions.json",
            mime="application/json",
            use_container_width=True,
        )


def _clear_quiz_state():
    keys = [
        "quiz_active", "quiz_done", "quiz_state", "quiz_result",
        "pool_by_tier", "pdf_pages", "pdf_name", "difficulty_mode",
    ]
    for k in keys:
        st.session_state.pop(k, None)


# ============== MAIN ==============

def main():
    if not st.session_state.get("logged_in"):
        render_login_page()
    elif st.session_state.get("quiz_active"):
        render_quiz()
    elif st.session_state.get("quiz_done"):
        render_results()
    else:
        render_home()

main()
