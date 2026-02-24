import streamlit as st
import sqlite3
import json
import random
import re
import requests
from datetime import datetime

import fitz  # PyMuPDF
from fpdf import FPDF

# Gemini API Configuration
API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"

# ============== DATABASE SETUP (all in one file) ==============

DB_PATH = "ascendquiz.db"

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
    
    c.execute('''CREATE TABLE IF NOT EXISTS quiz_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        pdf_name TEXT,
        questions JSON,
        final_score INTEGER,
        total_questions_answered INTEGER,
        mastery_achieved BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        question_text TEXT,
        difficulty INTEGER,
        correct BOOLEAN,
        topic TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES quiz_sessions(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS topic_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT,
        attempts INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        last_attempted TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        UNIQUE(user_id, topic)
    )''')
    
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

def save_quiz_session(user_id, pdf_name, questions, final_score, total_answered, mastery_achieved):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO quiz_sessions 
                 (user_id, pdf_name, questions, final_score, total_questions_answered, mastery_achieved)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (user_id, pdf_name, json.dumps(questions), final_score, total_answered, mastery_achieved))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

def save_response(session_id, question_text, difficulty, correct, topic=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO responses (session_id, question_text, difficulty, correct, topic)
                 VALUES (?, ?, ?, ?, ?)''',
              (session_id, question_text, difficulty, correct, topic))
    conn.commit()
    conn.close()

def update_topic_stats(user_id, topic, correct):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO topic_stats (user_id, topic, attempts, correct, last_attempted)
                 VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP)
                 ON CONFLICT(user_id, topic) DO UPDATE SET
                 attempts = attempts + 1,
                 correct = correct + ?,
                 last_attempted = CURRENT_TIMESTAMP''',
              (user_id, topic, 1 if correct else 0, 1 if correct else 0))
    conn.commit()
    conn.close()

def get_user_history(user_id):
    """Get quiz history excluding demo quizzes."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''SELECT id, pdf_name, final_score, total_questions_answered, mastery_achieved, created_at
                 FROM quiz_sessions
                 WHERE user_id = ? AND pdf_name != 'Demo Quiz'
                 ORDER BY created_at DESC''', (user_id,))
    sessions = [dict(row) for row in c.fetchall()]
    conn.close()
    return sessions

def get_user_stats(user_id):
    """Get user statistics excluding demo quizzes."""
    conn = get_connection()
    c = conn.cursor()

    # Exclude demo quizzes from stats
    c.execute('''SELECT
                    COUNT(*) as total_quizzes,
                    AVG(final_score) as avg_score,
                    SUM(CASE WHEN mastery_achieved THEN 1 ELSE 0 END) as mastery_count,
                    SUM(total_questions_answered) as total_questions
                 FROM quiz_sessions
                 WHERE user_id = ? AND pdf_name != 'Demo Quiz' ''', (user_id,))
    overall = dict(c.fetchone())

    c.execute('''SELECT topic, attempts, correct,
                    ROUND(correct * 100.0 / attempts, 1) as accuracy
                 FROM topic_stats WHERE user_id = ? ORDER BY attempts DESC''', (user_id,))
    topics = [dict(row) for row in c.fetchall()]

    # Exclude demo quizzes from recent performance
    c.execute('''SELECT final_score, created_at FROM quiz_sessions
                 WHERE user_id = ? AND pdf_name != 'Demo Quiz'
                 ORDER BY created_at DESC LIMIT 5''', (user_id,))
    recent = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return {"overall": overall, "topics": topics, "recent": recent}

def get_weak_topics(user_id, threshold=60):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''SELECT topic, attempts, correct,
                    ROUND(correct * 100.0 / attempts, 1) as accuracy
                 FROM topic_stats 
                 WHERE user_id = ? AND attempts >= 2
                 AND (correct * 100.0 / attempts) < ?
                 ORDER BY accuracy ASC''', (user_id, threshold))
    weak = [dict(row) for row in c.fetchall()]
    conn.close()
    return weak

# Initialize DB
init_db()

# ============== PDF PROCESSING ==============

def extract_text_from_pdf(pdf_file):
    """Extract text from each page of a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def get_chunks_by_token(pages):
    """
    Chunks the extracted PDF text based on a 10,000 token limit per chunk.
    - If total tokens <= 10k, it returns one chunk.
    - If total tokens <= 20k, it returns two chunks.
    - If total tokens > 20k, it randomly selects two chunks.
    """
    full_text = "\n\n".join(pages)
    TOKEN_CHUNK_SIZE = 10000 * 4  # ~40,000 characters per chunk (1 token ≈ 4 chars)

    all_text_chunks = [full_text[i:i + TOKEN_CHUNK_SIZE] for i in range(0, len(full_text), TOKEN_CHUNK_SIZE)]
    num_chunks = len(all_text_chunks)

    if num_chunks <= 2:
        return all_text_chunks
    else:
        return random.sample(all_text_chunks, 2)

# ============== QUESTION GENERATION ==============

def generate_prompt(text_chunk):
    """Generate the detailed pedagogical prompt for Gemini to create quiz questions."""
    return f"""
You are a teacher who is designing a test with multiple choice questions (each with 4 answer choices) to test content from a passage.
Given the following passage or notes, generate exactly 30 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 20 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage.
**CRITICAL REQUIREMENT - NO TEXT REFERENCES:**
- Questions must be COMPLETELY SELF-CONTAINED and not reference the original text
- DO NOT use phrases like "according to the passage," "the text states," "the first example," "as mentioned," "the author discusses," etc.
- DO NOT reference specific figures, tables, pages, or sections from the passage
- Present all necessary context within the question itself
- Students should be able to answer based on their understanding of the concepts, not memory of where things appeared in the text
- Frame questions as direct concept tests, not reading comprehension
- If there is information about ISBN or ebook distribution consequences or copyrights, do not ask questions about these things. Only ask questions about academic content
**CRITICAL: Design Questions That Test TRUE MASTERY, Not Test-Taking Skills**
Your goal is to create questions where students CANNOT get the correct answer through:
- Process of elimination with obviously implausible answers
- Common sense reasoning without domain-specific knowledge
- Guessing based on option patterns, lengths, or complexity differences
- Recognizing what "sounds right" based on everyday language
- Using the question wording itself as a hint to the answer
- For questions above the remember difficulty band, questions that a student who memorizes information in the reading without true understanding can answer correctly
Generate exactly 30 questions that vary across difficulty levels. Questions should test **conceptual understanding and application**, not just recall of text. Use the uploaded material to determine:
1. What concepts are explicitly stated and factual: these support easy or "Remember" questions.
2. What concepts require connecting multiple ideas or interpreting examples: these support medium or **Understand** or **Apply** questions.
3. What concepts require analysis of interactions, synthesis, or predicting outcomes based on material in the text → these support medium/hard and hard or **Analyze**, **Evaluate**, or **Create** questions.
Use the passage to determine which concepts can be recalled, applied, analyzed, or synthesized. Do not assign difficulty randomly.
**For EVERY question, ensure:**
1. **All four options are plausible to someone WITHOUT domain expertise**
   - Wrong answers should represent actual misconceptions or partial understanding
   - Avoid absurd options that anyone could eliminate (e.g., if asking about a biological process, don't include "it turns purple" as an option)
   - All options should be similar in length, specificity, and technical complexity
   - Don't mix highly technical language in one option with casual language in others
2. **The question cannot be answered through linguistic/semantic clues alone**
   - Don't ask "What does [term] do?" when the term's name in everyday English reveals the answer
   - Avoid questions where the correct answer repeats key words from the question
   - Don't make the correct answer significantly more detailed/specific than wrong answers
   - Ensure wrong answers use equally precise terminology
3. **Wrong answers reflect genuine confusion, not nonsense**
   - Each wrong answer should be what a student might choose if they:
     * Confused two related concepts
     * Applied a rule from a different context
     * Made a common calculation error
     * Remembered only part of the concept
   - Never include options that are absurd or completely unrelated to the topic
**Difficulty Calibration Guidelines:**
When estimating "estimated_correct_pct", consider that students may have:
- General intelligence and test-taking skills
- Ability to eliminate absurd options
- Common sense reasoning
- Pattern recognition abilities
**Your difficulty estimates should reflect:**
85–100% correct (Very Easy / Direct Recall)
Students can answer by recalling a fact, definition, or formula explicitly stated in the passage.
Requires no calculation, inference, or application beyond what is written.
All four options must be plausible and technically correct; distractors should reflect common small misconceptions.
70–84% correct (Easy / Understanding / Single-Step Reasoning)
Requires one step of reasoning or minor inference beyond direct recall.
Students must connect a concept in the passage to a similar context or slightly different phrasing, but it is still straightforward.
Wrong answers should reflect adjacent or related concepts that a partial understanding might confuse.
50–69% correct (Medium / Application / Multi-Step Reasoning)
Requires applying principles from the passage to a new scenario or combining multiple pieces of information.
The passage does not explicitly solve this problem, so students must adapt knowledge.
Distractors should be plausible errors that someone might make if they misapplied formulas, misremembered conditions, or partially understood the concept.
30–49% correct (Hard / Analysis / Synthesis)
Requires deep understanding, integration, or analysis of multiple concepts in the passage.
Students must infer relationships, compare methods, or predict outcomes not directly explained.
Wrong answers should seem correct to someone with partial understanding, exploiting subtle distinctions or counterintuitive results.
Below 30% correct (Very Hard / Evaluation or Creation)
Requires expert-level judgment, design, or synthesis, combining multiple principles in novel ways.
Multiple answers might seem defensible; students must evaluate, critique, or generate solutions based on passage principles.
Distractors reflect plausible alternative interpretations, partial understanding, or common advanced mistakes.
**Requirements**:
- 8 easy (≥85%), 7 medium (60–84%), 8 medium-hard (40-60%), 7 hard (<40%)
**Each question must include the following fields:**
- "question": A clear, concise, and unambiguous question that tests understanding of concepts from the passage. The question should be COMPLETELY SELF-CONTAINED with all necessary context included. Never reference "the passage," "the text," specific examples by position (first, second, etc.), or figures/tables. Ask about the concept directly.
- "options": An array of exactly 4 strings in this exact format:
    [
      "A. [First option text]",
      "B. [Second option text]",
      "C. [Third option text]",
      "D. [Fourth option text]"
    ]
  Each string must start with the letter and period. Do not use an object/dictionary structure. It is 4 plausible answer choices labeled "A", "B", "C", and "D" (with one being correct). ALL four options must be similar in:
    * Length (within 20% of each other)
    * Specificity and detail level
    * Technical complexity
    * Grammatical structure
  Wrong answers must represent genuine misconceptions from the domain, not random nonsense.
- "correct_answer": The letter ("A", "B", "C", or "D") corresponding to the correct option.
- "explanation": A deep, pedagogically useful explanation that teaches the concept behind the correct answer. The explanation must:
    1. Start by stating the correct letter and full answer
    2. Explain WHY that answer is correct using conceptual reasoning - explain mechanisms, properties, or principles
    3. For each incorrect answer, explain:
       - Why it's wrong
       - What specific misconception or error would lead someone to choose it
       - What partial understanding might make it seem correct
    4. Focus on teaching the underlying concept, not referencing where information appeared in the text
    5. Use the tone of a tutor helping a student understand the concept
- "cognitive_level": Choose from "Remember", "Understand", "Apply", "Analyze", "Evaluate", or "Create" based on the cognitive skill actually tested.
- "estimated_correct_pct": Numeric estimate of percentage of students expected to answer correctly (0-100).
  **CRITICAL**: If your estimate is below 70%, you MUST verify:
  - All four options are genuinely plausible to a non-expert
  - No options can be eliminated through pure logic/common sense
  - The question cannot be answered by someone clever who lacks domain knowledge
  - Wrong answers represent actual conceptual confusions, not absurdities
  If you cannot verify all of these, INCREASE the percentage estimate.
- "reasoning": Brief rationale for the percentage assignment. **If estimated_correct_pct < 70%**, you MUST explain:
  1. What specific domain knowledge is required that common sense/logic cannot provide
  2. Why each wrong answer would seem plausible to someone with partial understanding
  3. What makes this question resistant to test-taking strategies
  If you cannot provide specific explanations for all three points, your difficulty estimate is too low.
All math expressions must use valid LaTeX format with $...$ for inline math and $$...$$ for display math.
Before finalizing each question, verify that the correct answer and every explanation are explicitly supported by factual information or definitions present in the passage. Please make sure that every correct answer is clearly correct and every incorrect answer is clearly incorrect.
Focus on testing conceptual understanding rather than text memorization.
If the passage contains code, mathematical derivations, or data tables, generate questions about:
- How the logic/process works (not "what does line 5 do")
- What results mean and why (not "what is the output")
- When to apply methods (not "what is this method called")
- Why approaches differ (not "which method is shown")
Return **only** a valid JSON array of 30 questions. Focus on testing conceptual understanding rather than text memorization.
Do not include any text, commentary, or markdown fences.
Output must begin with `[` and end with `]` — no explanations outside JSON.
Passage:
{text_chunk}
"""

def call_gemini_api(prompt):
    """Call the Gemini API with the given prompt."""
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 20000
        }
    }
    url = f"{GEMINI_URL}?key={API_KEY}"
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        return None, response.text
    response_json = response.json()
    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"], None
    except (KeyError, IndexError) as e:
        return None, f"Failed to parse Gemini API response: {str(e)}"

def clean_response_text(text: str) -> str:
    """Extract JSON from model response, stripping markdown fences and commentary."""
    text = text.strip()
    fence_patterns = [
        r"```json\s*(.*?)```",
        r"```\s*(.*?)```",
        r"`{3,}\s*json\s*(.*?)`{3,}",
        r"`{3,}\s*(.*?)`{3,}"
    ]

    for pattern in fence_patterns:
        fence_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
            break

    # Find JSON array boundaries
    start_idx = text.find('[')
    end_idx = text.rfind(']')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1].strip()

    # Fallback: object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1].strip()

    return text

def repair_json(text: str) -> str:
    """Repair malformed or truncated JSON from model output."""
    text = re.sub(r'```(?:json)?', '', text)
    text = text.replace('```', '').strip()

    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        text = text[start:end + 1]
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = f"[{text[start:end + 1]}]"

    # Fix incomplete trailing objects
    open_braces = text.count('{')
    close_braces = text.count('}')
    if close_braces < open_braces:
        last_full = text.rfind('}')
        if last_full != -1:
            text = text[:last_full + 1]
        text += "]" if not text.endswith(']') else ""

    text = re.sub(r'}\s*{', '}, {', text)
    text = re.sub(r',\s*([\]}])', r'\1', text)

    if not text.startswith('['):
        text = '[' + text
    if not text.endswith(']'):
        text = text + ']'

    return text.strip()

def parse_question_json(text: str):
    """Parse JSON with multiple fallback strategies."""
    cleaned = clean_response_text(text)
    cleaned = repair_json(cleaned)

    # Try standard JSON parsing
    try:
        result = json.loads(cleaned)
        return result
    except json.JSONDecodeError as e:
        # Try json5 as fallback
        try:
            import json5
            result = json5.loads(cleaned)
            return result
        except Exception as e2:
            # Final fallback: extract individual questions manually
            try:
                questions = []
                question_pattern = r'\{\s*"question":[^}]*?"reasoning":[^}]*?\}'
                potential_questions = re.findall(question_pattern, cleaned, re.DOTALL)

                for q_text in potential_questions:
                    try:
                        q_obj = json.loads(q_text)
                        questions.append(q_obj)
                    except:
                        continue

                if questions:
                    return questions
            except:
                pass

            return []

def filter_invalid_difficulty_alignment(questions):
    """Filter questions where cognitive level doesn't align with estimated difficulty."""
    bloom_difficulty_ranges = {
        "Remember": (80, 100),
        "Understand": (50, 90),
        "Apply": (45, 80),
        "Analyze": (25, 65),
        "Evaluate": (0, 60),
        "Create": (0, 50)
    }
    valid = []
    invalid = []
    for q in questions:
        if not isinstance(q, dict):
            invalid.append(q)
            continue
        cog = str(q.get("cognitive_level", "")).strip().capitalize()
        try:
            pct = int(q.get("estimated_correct_pct", -1))
        except Exception:
            pct = -1
        if cog in bloom_difficulty_ranges and 0 <= pct <= 100:
            low, high = bloom_difficulty_ranges[cog]
            if low <= pct <= high:
                valid.append(q)
            else:
                invalid.append(q)
        else:
            invalid.append(q)
    return valid, invalid

# ============== PERFORMANCE SUMMARY & REPORTS ==============

def generate_performance_summary(answers):
    """
    Analyzes quiz performance data and generates a pedagogical summary using Gemini.
    'answers' is a list of tuples: (difficulty, correctness, question_obj)
    """
    performance_data = []
    for item in answers:
        # Handle both 2-tuple and 3-tuple formats
        if len(item) == 3:
            diff, correct, q = item
        else:
            diff, correct = item
            q = {"question": "N/A", "cognitive_level": "N/A", "explanation": "N/A"}

        status = "Correct" if correct else "Incorrect"
        q_text = q.get("question", "N/A") if isinstance(q, dict) else "N/A"
        cog_level = q.get("cognitive_level", "N/A") if isinstance(q, dict) else "N/A"
        explanation = q.get("explanation", "N/A")[:200] if isinstance(q, dict) else "N/A"

        performance_data.append(
            f"- Question: {q_text}\n"
            f"  Topic/Level: {cog_level}\n"
            f"  Result: {status}\n"
            f"  Explanation provided: {explanation}..."
        )

    performance_string = "\n".join(performance_data)

    summary_prompt = f"""
    You are an expert educational consultant and tutor. You are reviewing a student's performance on a computer-adaptive mastery quiz.

    Below is a list of questions the student answered, their cognitive level, and whether the student got them right or wrong:

    {performance_string}

    Based ONLY on this performance data, provide a comprehensive "Learning Progress Report" with the following sections:

    1. **Overall Performance Evaluation**: A 2-3 sentence encouraging summary of their current mastery level.
    2. **Key Strengths (Top 3)**: Identify specific concepts or cognitive skills (e.g., "Application of formulas", "Conceptual understanding of X") the student has mastered.
    3. **Growth Areas (Top 2)**: Identify specific gaps in knowledge or reasoning where the student struggled.
    4. **Actionable Study Plan**: Provide 3 specific, concrete steps the student should take to improve their understanding of the material.

    **Tone**: Professional, encouraging, and highly specific to the content mentioned in the questions.
    **Format**: Use Markdown for headers and bullet points.
    """

    summary_text, error = call_gemini_api(summary_prompt)

    if error:
        return f"Could not generate summary: {error}"
    return summary_text

def create_pdf_report(summary_text, mastery_score, missed_questions=None):
    """Generate a PDF report with AI summary and missed questions review."""
    pdf = FPDF()
    pdf.add_page()

    # Helper to clean text for PDF
    def clean(text):
        if text is None:
            return ""
        return str(text).encode('latin-1', 'ignore').decode('latin-1')

    # === HEADER ===
    pdf.set_font("helvetica", 'B', 20)
    pdf.cell(0, 15, txt="AscendQuiz", ln=True, align='C')
    pdf.set_font("helvetica", 'I', 12)
    pdf.cell(0, 8, txt="Learning Progress Report", ln=True, align='C')
    pdf.ln(5)

    # === MASTERY SCORE BOX ===
    pdf.set_fill_color(102, 126, 234)  # Purple
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 12, txt=f"  Final Mastery Score: {mastery_score}%", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # === AI SUMMARY SECTION ===
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, txt="AI Performance Analysis", ln=True)
    pdf.set_draw_color(102, 126, 234)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("helvetica", size=10)
    clean_summary = clean(summary_text)
    pdf.multi_cell(0, 6, txt=clean_summary)
    pdf.ln(10)

    # === MISSED QUESTIONS REVIEW ===
    if missed_questions and len(missed_questions) > 0:
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 14)
        pdf.cell(0, 10, txt=f"Questions to Review ({len(missed_questions)} missed)", ln=True)
        pdf.set_draw_color(220, 53, 69)  # Red
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

        for i, (difficulty, correct, q) in enumerate(missed_questions, 1):
            # Question number and difficulty
            pdf.set_fill_color(248, 249, 250)
            pdf.set_font("helvetica", 'B', 11)
            pdf.cell(0, 8, txt=f"  Question {i} (Difficulty: {difficulty}/8)", ln=True, fill=True)
            pdf.ln(3)

            # Question text
            pdf.set_font("helvetica", 'B', 10)
            pdf.multi_cell(0, 6, txt=clean(q.get("question", "N/A")))
            pdf.ln(3)

            # Options
            pdf.set_font("helvetica", size=9)
            correct_letter = q.get("correct_answer", "").strip().upper()
            options = q.get("options", [])

            for opt in options:
                opt_text = clean(opt)
                # Highlight correct answer
                if opt_text.startswith(correct_letter + ".") or opt_text.startswith(correct_letter + " "):
                    pdf.set_text_color(40, 167, 69)  # Green
                    pdf.multi_cell(0, 5, txt=f"  [CORRECT] {opt_text}")
                    pdf.set_text_color(0, 0, 0)
                else:
                    pdf.multi_cell(0, 5, txt=f"  {opt_text}")

            pdf.ln(3)

            # Explanation
            pdf.set_font("helvetica", 'I', 9)
            pdf.set_text_color(80, 80, 80)
            explanation = clean(q.get("explanation", "No explanation available."))
            pdf.multi_cell(0, 5, txt=f"Explanation: {explanation}")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(8)

            # Add page break if needed
            if pdf.get_y() > 250:
                pdf.add_page()

    # === FOOTER ===
    pdf.set_y(-20)
    pdf.set_font("helvetica", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, txt="Generated by AscendQuiz - Your AI-Powered Learning Assistant", align='C')

    return pdf.output()

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="AscendQuiz", page_icon="📚", layout="wide")

# ============== SAMPLE QUESTIONS (NO API NEEDED) ==============
DEMO_QUESTIONS = [
    {
        "question": "What is the primary function of mitochondria in a cell?",
        "options": ["A. Protein synthesis", "B. ATP production through cellular respiration", "C. DNA replication", "D. Cell membrane formation"],
        "correct_answer": "B",
        "explanation": "Mitochondria are the 'powerhouses' of the cell, responsible for producing ATP through cellular respiration. Protein synthesis occurs in ribosomes, DNA replication in the nucleus, and cell membrane formation involves the endoplasmic reticulum and Golgi apparatus.",
        "topic": "Cell Biology",
        "estimated_correct_pct": 85
    },
    {
        "question": "In the equation y = mx + b, what does 'm' represent?",
        "options": ["A. Y-intercept", "B. X-intercept", "C. Slope of the line", "D. Origin point"],
        "correct_answer": "C",
        "explanation": "In slope-intercept form (y = mx + b), 'm' represents the slope (rate of change) and 'b' represents the y-intercept. The slope tells you how much y changes for each unit increase in x.",
        "topic": "Linear Equations",
        "estimated_correct_pct": 90
    },
    {
        "question": "Which data structure uses LIFO (Last In, First Out) ordering?",
        "options": ["A. Queue", "B. Stack", "C. Linked List", "D. Binary Tree"],
        "correct_answer": "B",
        "explanation": "A stack follows LIFO ordering - the last element added is the first one removed (like a stack of plates). Queues use FIFO (First In, First Out). Linked lists and binary trees don't have inherent LIFO/FIFO ordering.",
        "topic": "Data Structures",
        "estimated_correct_pct": 75
    },
    {
        "question": "What is the derivative of f(x) = x³?",
        "options": ["A. 3x", "B. x²", "C. 3x²", "D. x⁴/4"],
        "correct_answer": "C",
        "explanation": "Using the power rule: d/dx[xⁿ] = nxⁿ⁻¹. So d/dx[x³] = 3x³⁻¹ = 3x². Option A forgets to reduce the exponent, B forgets the coefficient, and D is the integral, not derivative.",
        "topic": "Calculus",
        "estimated_correct_pct": 70
    },
    {
        "question": "Which sorting algorithm has the best average-case time complexity?",
        "options": ["A. Bubble Sort - O(n²)", "B. Quick Sort - O(n log n)", "C. Selection Sort - O(n²)", "D. Insertion Sort - O(n²)"],
        "correct_answer": "B",
        "explanation": "Quick Sort has an average time complexity of O(n log n), which is significantly better than the O(n²) algorithms. Bubble, Selection, and Insertion sort all have quadratic time complexity on average.",
        "topic": "Algorithms",
        "estimated_correct_pct": 55
    },
    {
        "question": "In photosynthesis, where does the light-dependent reaction occur?",
        "options": ["A. Stroma", "B. Thylakoid membrane", "C. Mitochondria", "D. Cell wall"],
        "correct_answer": "B",
        "explanation": "Light-dependent reactions occur in the thylakoid membrane where chlorophyll absorbs light energy. The stroma is where the Calvin cycle (light-independent reactions) occurs. Mitochondria handle cellular respiration, not photosynthesis.",
        "topic": "Cell Biology",
        "estimated_correct_pct": 60
    },
    {
        "question": "What is the time complexity of binary search?",
        "options": ["A. O(n)", "B. O(n²)", "C. O(log n)", "D. O(1)"],
        "correct_answer": "C",
        "explanation": "Binary search divides the search space in half each iteration, resulting in O(log n) complexity. Linear search is O(n), and O(1) would mean constant time regardless of input size, which isn't possible for searching.",
        "topic": "Algorithms",
        "estimated_correct_pct": 65
    },
    {
        "question": "Which HTTP method is idempotent and used for completely replacing a resource?",
        "options": ["A. POST", "B. PUT", "C. PATCH", "D. GET"],
        "correct_answer": "B",
        "explanation": "PUT is idempotent (multiple identical requests have the same effect as one) and replaces the entire resource. POST creates new resources and isn't idempotent. PATCH partially updates. GET only retrieves data.",
        "topic": "Web Development",
        "estimated_correct_pct": 45
    },
    {
        "question": "What is the value of log₂(8)?",
        "options": ["A. 2", "B. 3", "C. 4", "D. 8"],
        "correct_answer": "B",
        "explanation": "log₂(8) asks '2 raised to what power equals 8?' Since 2³ = 8, the answer is 3. This is fundamental to understanding logarithms as the inverse of exponentiation.",
        "topic": "Logarithms",
        "estimated_correct_pct": 80
    },
    {
        "question": "In object-oriented programming, what is encapsulation?",
        "options": ["A. Creating multiple instances of a class", "B. Hiding internal state and requiring interaction through methods", "C. Inheriting properties from a parent class", "D. Defining multiple methods with the same name"],
        "correct_answer": "B",
        "explanation": "Encapsulation bundles data and methods together while hiding internal state, only allowing access through defined interfaces. Option A describes instantiation, C describes inheritance, and D describes method overloading.",
        "topic": "OOP Concepts",
        "estimated_correct_pct": 70
    },
    {
        "question": "What is the integral of 2x?",
        "options": ["A. x²", "B. x² + C", "C. 2", "D. 2x²"],
        "correct_answer": "B",
        "explanation": "The integral of 2x is x² + C, where C is the constant of integration. When integrating, we add 1 to the exponent and divide by the new exponent: 2x¹ → 2x²/2 = x². The +C is essential for indefinite integrals.",
        "topic": "Calculus",
        "estimated_correct_pct": 65
    },
    {
        "question": "Which of the following is NOT a valid Python data type?",
        "options": ["A. list", "B. tuple", "C. array", "D. dictionary"],
        "correct_answer": "C",
        "explanation": "Python has built-in types: list, tuple, dict (dictionary), set, etc. 'array' is not a built-in type - you need to import it from the array module or use NumPy. Lists are typically used instead.",
        "topic": "Python Basics",
        "estimated_correct_pct": 55
    },
    {
        "question": "What is the pH of a neutral solution at 25°C?",
        "options": ["A. 0", "B. 7", "C. 14", "D. 1"],
        "correct_answer": "B",
        "explanation": "A neutral solution has equal concentrations of H⁺ and OH⁻ ions, which at 25°C corresponds to pH 7. pH below 7 is acidic, above 7 is basic. pH 0 and 1 are very acidic, pH 14 is very basic.",
        "topic": "Chemistry",
        "estimated_correct_pct": 85
    },
    {
        "question": "In a relational database, what does SQL stand for?",
        "options": ["A. Simple Query Language", "B. Structured Query Language", "C. Standard Question Language", "D. System Query Logic"],
        "correct_answer": "B",
        "explanation": "SQL stands for Structured Query Language. It's the standard language for managing and manipulating relational databases, used for queries, updates, insertions, and deletions.",
        "topic": "Databases",
        "estimated_correct_pct": 80
    },
    {
        "question": "What is the Big O complexity of accessing an element by index in an array?",
        "options": ["A. O(n)", "B. O(log n)", "C. O(1)", "D. O(n²)"],
        "correct_answer": "C",
        "explanation": "Array access by index is O(1) - constant time - because arrays store elements in contiguous memory. The memory address can be calculated directly: base_address + (index × element_size), regardless of array size.",
        "topic": "Data Structures",
        "estimated_correct_pct": 60
    }
]

# ============== ADAPTIVE QUIZ LOGIC ==============

def assign_difficulty_label(estimated_pct):
    """Map estimated correctness percentage to difficulty tier (1-8)."""
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
    """Organize questions into 8 difficulty tiers."""
    groups = {i: [] for i in range(1, 9)}
    for q in questions:
        pct = q.get("estimated_correct_pct", 0)
        label = assign_difficulty_label(pct)
        if label:
            q["difficulty_label"] = label
            groups[label].append(q)
    return groups

def pick_question(diff, asked, all_qs):
    """Get available questions at a difficulty level that haven't been asked."""
    pool = all_qs.get(diff, [])
    return [(i, q) for i, q in enumerate(pool) if (diff, i) not in asked]

def find_next_difficulty(current_diff, going_up, asked, all_qs):
    """Find the next difficulty tier with available questions."""
    next_diff = current_diff + 1 if going_up else current_diff - 1
    if 1 <= next_diff <= 8 and pick_question(next_diff, asked, all_qs):
        return next_diff
    search_range = (
        range(next_diff + 1, 9) if going_up else range(next_diff - 1, 0, -1)
    )
    for d in search_range:
        if pick_question(d, asked, all_qs):
            return d
    return current_diff

def get_next_question(current_diff, asked, all_qs):
    """Select a random question from the current difficulty tier."""
    available = pick_question(current_diff, asked, all_qs)
    if not available:
        return current_diff, None, None
    idx, q = random.choice(available)
    return current_diff, idx, q

def accuracy_on_levels(answers, levels):
    """Calculate accuracy for specific difficulty levels."""
    filtered = [item[1] for item in answers if item[0] in levels]
    return sum(filtered) / len(filtered) if filtered else 0


## REMOVED CONFIDENCE SCALING AND ADDED A PER-QUESTION ACCUMULATION SYSTEM (w/ penalties)
def compute_mastery_score(answers):
    """
    Compute mastery score via per-question point accumulation.
    Correct answers add points based on difficulty; wrong answers deduct
    a fraction of those points. Score is clamped to [0, 100].
    Mastery threshold is 70.
    """
    if not answers:
        return 0

    BAND_POINTS = {
        (1, 2): 5,    # Easy
        (3, 4): 7,    # Medium
        (5, 6): 9,    # Medium-Hard
        (7, 8): 10,   # Hard
    }
    WRONG_PENALTY_FACTOR = 1 / 3

    raw_score = 0.0
    for item in answers:
        difficulty = item[0]
        correct = item[1]
        for levels, points in BAND_POINTS.items():
            if difficulty in levels:
                if correct:
                    raw_score += points
                else:
                    raw_score -= points * WRONG_PENALTY_FACTOR
                break

    return int(round(max(0.0, min(100.0, raw_score))))

# ============== UI COMPONENTS ==============

def render_login_page():
    st.title("📚 AscendQuiz")
    st.markdown("### Your Personal Adaptive Learning Platform")
    
    st.markdown("""
    **Features:**
    - 🎯 Adaptive quizzes that adjust to your level
    - 📊 Track your progress over time  
    - 🔍 Identify weak topics to focus on
    - 📜 Review past quiz performance
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔑 Returning User")
        login_user = st.text_input("Username", key="login_username", placeholder="Enter your username")
        if st.button("Login", key="login_btn", use_container_width=True):
            if login_user:
                user = get_user(login_user)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ User not found. Please sign up first.")
            else:
                st.warning("Please enter a username")
    
    with col2:
        st.markdown("#### ✨ New User")
        new_user = st.text_input("Choose a Username", key="signup_username", placeholder="Pick a username")
        if st.button("Create Account", key="signup_btn", use_container_width=True):
            if new_user:
                if len(new_user) < 3:
                    st.error("Username must be at least 3 characters")
                else:
                    user_id, error = create_user(new_user)
                    if user_id:
                        st.session_state.user = {"id": user_id, "username": new_user}
                        st.session_state.logged_in = True
                        st.success("✅ Account created!")
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
            else:
                st.warning("Please choose a username")

def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user['username']}")
        
        stats = get_user_stats(st.session_state.user["id"])
        if stats["overall"]["total_quizzes"]:
            st.metric("Quizzes Taken", stats["overall"]["total_quizzes"])
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📝 Take Quiz", "📊 Dashboard", "📜 History"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        return page

def render_dashboard():
    st.title("📊 Your Learning Dashboard")
    
    user_id = st.session_state.user["id"]
    stats = get_user_stats(user_id)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Quizzes Taken", stats["overall"]["total_quizzes"] or 0)
    with col2:
        avg = stats["overall"]["avg_score"]
        st.metric("Avg Score", f"{avg:.0f}%" if avg else "N/A")
    with col3:
        st.metric("Mastery Achieved", stats["overall"]["mastery_count"] or 0)
    with col4:
        st.metric("Questions Answered", stats["overall"]["total_questions"] or 0)

    st.markdown("### 📈 Recent Performance")
    if stats["recent"]:
        for session in stats["recent"]:
            score = session["final_score"] or 0
            date = session["created_at"][:10] if session["created_at"] else "Unknown"

            bar_color = "#28a745" if score >= 70 else "#ffc107" if score >= 50 else "#dc3545"
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 10px 15px; border-radius: 8px; margin: 5px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span><strong>{score}%</strong></span>
                    <span style="color: #888;">{date}</span>
                </div>
                <div style="background: #e9ecef; border-radius: 4px; height: 8px;">
                    <div style="background: {bar_color}; width: {score}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📝 No quizzes completed yet. Start one now!")

def render_history():
    st.title("📜 Quiz History")
    
    user_id = st.session_state.user["id"]
    history = get_user_history(user_id)
    
    if not history:
        st.info("📝 No quizzes completed yet. Take your first quiz!")
        return
    
    st.markdown(f"**Total quizzes:** {len(history)}")
    st.markdown("---")
    
    for i, session in enumerate(history):
        score = session['final_score'] or 0
        status_color = "#28a745" if session['mastery_achieved'] else "#ffc107"
        status_text = "✅ Mastered" if session['mastery_achieved'] else "📖 In Progress"
        
        with st.expander(f"📄 {session['pdf_name'] or 'Demo Quiz'} — Score: {score}% — {session['created_at'][:10] if session['created_at'] else 'Unknown'}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Score", f"{score}%")
            with col2:
                st.metric("Questions Answered", session['total_questions_answered'] or 0)
            with col3:
                st.markdown(f"**Status**")
                st.markdown(f"<span style='color: {status_color}; font-size: 1.2em;'>{status_text}</span>", 
                           unsafe_allow_html=True)

def render_quiz():
    st.title("📝 Take a Quiz")

    # Loading tips to show during question generation
    LOADING_TIPS = [
        "💡 The quiz adapts to your level - harder questions unlock higher scores!",
        "🎯 Mastery is achieved at 70% - answer harder questions correctly to get there faster.",
        "📚 Questions are generated using Bloom's Taxonomy for varied cognitive levels.",
        "⚡ The difficulty adjusts after each answer based on your performance.",
        "🧠 Each question tests conceptual understanding, not just memorization.",
    ]

    # Handle regeneration from existing PDF
    if st.session_state.get("regenerate_from_pdf", False) and "pdf_pages" in st.session_state:
        st.session_state.regenerate_from_pdf = False
        pdf_name = st.session_state.get("pdf_name", "Uploaded PDF")

        tip_placeholder = st.empty()
        progress_placeholder = st.empty()

        with st.spinner(""):
            tip_placeholder.info(f"💡 {random.choice(LOADING_TIPS)}")
            progress_placeholder.markdown(f"**Generating new questions from {pdf_name}...**")

            try:
                pages = st.session_state.pdf_pages
                chunks_to_use = get_chunks_by_token(pages)

                all_questions = []
                for i, chunk in enumerate(chunks_to_use):
                    if not chunk.strip():
                        continue
                    tip_placeholder.info(f"💡 {random.choice(LOADING_TIPS)}")
                    prompt = generate_prompt(chunk)
                    response_text, error = call_gemini_api(prompt)
                    if error:
                        st.error(f"API error: {error}")
                        # Clear regeneration state on error
                        for key in ["pdf_pages", "pdf_name"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.stop()
                    parsed = parse_question_json(response_text)
                    valid, invalid = filter_invalid_difficulty_alignment(parsed)
                    all_questions.extend(valid)

                tip_placeholder.info(f"💡 {random.choice(LOADING_TIPS)}")
                progress_placeholder.markdown("**Validating and organizing questions...**")

                if len(all_questions) < 20:
                    st.error(f"Only {len(all_questions)} valid questions generated. Need at least 20.")
                    for key in ["pdf_pages", "pdf_name"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.stop()

                # Cap pool at exactly 30, preserving difficulty spread
                if len(all_questions) > 30:
                    random.shuffle(all_questions)
                    all_questions = all_questions[:30]

                # Set up quiz state
                st.session_state.all_questions = all_questions
                st.session_state.questions_by_difficulty = group_by_difficulty(all_questions)
                st.session_state.quiz_mode = "pdf"
                st.session_state.quiz_active = True
                st.session_state.quiz_state = {
                    "current_difficulty": 4,
                    "asked": set(),
                    "answers": [],
                    "quiz_end": False,
                    "current_q_idx": None,
                    "current_q": None,
                    "show_explanation": False,
                    "last_correct": None,
                }

                st.session_state.current_session_id = save_quiz_session(
                    st.session_state.user["id"],
                    pdf_name,
                    all_questions,
                    0, 0, False
                )

                tip_placeholder.empty()
                progress_placeholder.empty()
                st.rerun()

            except Exception as e:
                st.error(f"Error regenerating questions: {str(e)}")
                for key in ["pdf_pages", "pdf_name"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.stop()

    # Check if we're in quiz mode
    if not st.session_state.get("quiz_active", False):
        # === PDF UPLOAD (PRIMARY) ===
        st.markdown("""
        ### Upload Your Study Material 📄

        Transform any PDF into an AI-generated adaptive quiz. The system will:
        - Extract key concepts from your document
        - Generate 30 questions across difficulty levels
        - Adapt the quiz to your performance in real-time

        *Processing takes 1-3 minutes depending on document length.*
        """)

        uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"], key="pdf_uploader")

        if uploaded_pdf:
            if st.button("🚀 Generate Quiz", use_container_width=True):
                # Show spinner with rotating tips
                tip_placeholder = st.empty()
                progress_placeholder = st.empty()

                with st.spinner(""):
                    tip_placeholder.info(f"💡 {random.choice(LOADING_TIPS)}")
                    progress_placeholder.markdown("**Step 1/3:** Extracting text from PDF...")

                    try:
                        pages = extract_text_from_pdf(uploaded_pdf)
                        st.session_state.pdf_pages = pages
                        st.session_state.pdf_name = uploaded_pdf.name

                        tip_placeholder.info(f"💡 {random.choice(LOADING_TIPS)}")
                        progress_placeholder.markdown("**Step 2/3:** Generating questions with AI...")

                        chunks_to_use = get_chunks_by_token(pages)

                        all_questions = []
                        for i, chunk in enumerate(chunks_to_use):
                            if not chunk.strip():
                                continue
                            tip_placeholder.info(f"💡 {random.choice(LOADING_TIPS)}")
                            prompt = generate_prompt(chunk)
                            response_text, error = call_gemini_api(prompt)
                            if error:
                                st.error(f"API error: {error}")
                                st.session_state.clear()
                                st.stop()
                            parsed = parse_question_json(response_text)
                            valid, invalid = filter_invalid_difficulty_alignment(parsed)
                            all_questions.extend(valid)

                        tip_placeholder.info(f"💡 {random.choice(LOADING_TIPS)}")
                        progress_placeholder.markdown("**Step 3/3:** Validating and organizing questions...")

                        # Check minimum question requirement
                        if len(all_questions) < 20:
                            st.error(f"Only {len(all_questions)} valid questions were generated. Need at least 20. Please try a different PDF.")
                            st.session_state.clear()
                            st.stop()

                        # Cap pool at exactly 30, preserving difficulty spread
                        if len(all_questions) > 30:
                            random.shuffle(all_questions)
                            all_questions = all_questions[:30]

                        # Set up quiz state
                        st.session_state.all_questions = all_questions
                        st.session_state.questions_by_difficulty = group_by_difficulty(all_questions)
                        st.session_state.quiz_mode = "pdf"
                        st.session_state.quiz_active = True
                        st.session_state.quiz_state = {
                            "current_difficulty": 4,
                            "asked": set(),
                            "answers": [],
                            "quiz_end": False,
                            "current_q_idx": None,
                            "current_q": None,
                            "show_explanation": False,
                            "last_correct": None,
                        }

                        # Save quiz session to database
                        st.session_state.current_session_id = save_quiz_session(
                            st.session_state.user["id"],
                            uploaded_pdf.name,
                            all_questions,
                            0, 0, False
                        )

                        tip_placeholder.empty()
                        progress_placeholder.empty()
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        st.session_state.clear()
                        st.stop()

        # === DEMO MODE (SECONDARY) ===
        st.markdown("---")
        st.markdown("#### Or try a demo first")
        st.caption("Preview how the adaptive quiz works with sample questions (no API needed).")

        if st.button("🎮 Try Demo Instead", use_container_width=False):
            questions = DEMO_QUESTIONS.copy()
            random.shuffle(questions)

            # Group demo questions by difficulty for adaptive logic
            questions_by_diff = group_by_difficulty(questions)

            st.session_state.all_questions = questions
            st.session_state.questions_by_difficulty = questions_by_diff
            st.session_state.quiz_mode = "demo"
            st.session_state.quiz_active = True
            st.session_state.quiz_state = {
                "current_difficulty": 4,
                "asked": set(),
                "answers": [],
                "quiz_end": False,
                "current_q_idx": None,
                "current_q": None,
                "show_explanation": False,
                "last_correct": None,
            }
            # No database session for demo mode
            st.session_state.current_session_id = None
            st.rerun()

        return

    # === ACTIVE QUIZ ===
    state = st.session_state.quiz_state
    all_qs = st.session_state.questions_by_difficulty
    is_pdf_mode = st.session_state.get("quiz_mode") == "pdf"

    # Check if quiz ended
    if state.get("quiz_end", False):
        render_quiz_complete()
        return

    # Get next question if needed
    if state["current_q"] is None and not state.get("show_explanation", False):
        diff, idx, q = get_next_question(state["current_difficulty"], state["asked"], all_qs)
        if q is None:
            state["quiz_end"] = True
            st.rerun()
        else:
            state["current_q"] = q
            state["current_q_idx"] = idx
            state["current_difficulty"] = diff

    # Calculate current score
    score = compute_mastery_score(state["answers"])
    num_answered = len(state["answers"])

    # Mastery progress bar
    mastery_color = "#28a745" if score >= 70 else "#ffc107" if score >= 50 else "#dc3545"
    st.markdown(f"""
    <div style="background: #e9ecef; border-radius: 10px; height: 30px; margin: 10px 0; position: relative;">
        <div style="background: {mastery_color}; width: {score}%; height: 100%; border-radius: 10px;"></div>
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 100%;
                    display: flex; align-items: center; justify-content: center; font-weight: bold;">
            Mastery: {score}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Questions answered: {num_answered}")

    q = state["current_q"]
    if q is None:
        state["quiz_end"] = True
        st.rerun()
        return

    difficulty = state["current_difficulty"]

    # Display question (don't show difficulty before answering)
    st.markdown(f"### {q['question']}")

    if not state.get("show_explanation", False):
        # Strip option labels for clean display
        def strip_leading_label(text):
            return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()

        option_labels = ["A", "B", "C", "D"]
        cleaned_options = [strip_leading_label(opt) for opt in q["options"]]
        rendered_options = []
        for label, text in zip(option_labels, cleaned_options):
            if "$" in text or "\\" in text:
                rendered_text = f"{label}. $${text}$$"
            else:
                rendered_text = f"{label}. {text}"
            rendered_options.append(rendered_text)

        selected = st.radio("Choose your answer:", options=rendered_options, key=f"q_{num_answered}", index=None)

        if st.button("Submit Answer", use_container_width=True):
            if selected is None:
                st.warning("Please select an answer!")
            else:
                selected_letter = selected.split(".")[0].strip().upper()
                correct_letter = q["correct_answer"].strip().upper()
                correct = (selected_letter == correct_letter)

                # Record answer
                state["asked"].add((difficulty, state["current_q_idx"]))
                state["answers"].append((difficulty, correct, q))
                state["last_correct"] = correct
                state["show_explanation"] = True

                # Save response to database (PDF mode only)
                if is_pdf_mode and st.session_state.current_session_id:
                    save_response(
                        st.session_state.current_session_id,
                        q["question"][:200],
                        difficulty,
                        correct,
                        q.get("cognitive_level", "General")
                    )

                # Check if mastery reached
                new_score = compute_mastery_score(state["answers"])
                if new_score >= 70:
                    state["quiz_end"] = True

                st.rerun()
    else:
        # Show answer feedback with difficulty revealed
        topic_or_level = q.get("topic", q.get("cognitive_level", "General"))
        st.markdown(f"""
        <div style="margin: 15px 0;">
            <span style="background: #e3f2fd; color: #1976d2; padding: 5px 12px; border-radius: 15px; font-size: 0.9em;">
                📚 {topic_or_level}
            </span>
            <span style="background: #fff3e0; color: #e65100; padding: 5px 12px; border-radius: 15px; font-size: 0.9em; margin-left: 8px;">
                ⚡ Level {difficulty}/8
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Get correct answer letter
        correct_letter = q["correct_answer"].strip().upper()

        # Simple result message
        if state["last_correct"]:
            st.success("✅ Correct! Great job!")
        else:
            st.error(f"❌ Incorrect. The correct answer was {correct_letter}.")

        # Simple explanation display
        st.info(f"**Explanation:** {q.get('explanation', 'No explanation available.')}")

        if st.button("Next Question →", use_container_width=True):
            # Adjust difficulty based on correctness
            if state["last_correct"]:
                state["current_difficulty"] = find_next_difficulty(
                    state["current_difficulty"], going_up=True, asked=state["asked"], all_qs=all_qs
                )
            else:
                state["current_difficulty"] = find_next_difficulty(
                    state["current_difficulty"], going_up=False, asked=state["asked"], all_qs=all_qs
                )

            state["current_q"] = None
            state["current_q_idx"] = None
            state["show_explanation"] = False
            state["last_correct"] = None
            st.rerun()


def render_quiz_complete():
    state = st.session_state.quiz_state
    answers = state["answers"]
    score = compute_mastery_score(answers)
    total = len(answers)
    correct_count = sum(1 for _, c, _ in answers if c) if answers and len(answers[0]) == 3 else sum(1 for _, c in answers if c)
    mastery = score >= 70
    is_pdf_mode = st.session_state.get("quiz_mode") == "pdf"

    # Update database (PDF mode only)
    if is_pdf_mode and st.session_state.get("current_session_id"):
        conn = get_connection()
        c = conn.cursor()
        c.execute('''UPDATE quiz_sessions SET final_score = ?, total_questions_answered = ?, mastery_achieved = ?
                    WHERE id = ?''', (score, total, mastery, st.session_state.current_session_id))
        conn.commit()
        conn.close()

    # Header
    st.markdown("## 🎉 Quiz Complete!")

    if mastery:
        st.balloons()
        st.success("🏆 Mastery Achieved! You've demonstrated strong understanding of the material.")
    else:
        st.warning("📖 Keep Practicing! You're making progress. Review the material and try again.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mastery Score", f"{score}%")
    with col2:
        st.metric("Questions Correct", f"{correct_count} / {total}")

    st.markdown("---")

    # Check for regeneration confirmation dialog
    if st.session_state.get("confirm_regenerate", False):
        pdf_name = st.session_state.get("pdf_name", "your PDF")
        st.warning(f"🔄 Generate new questions from **{pdf_name}**?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Generate New Questions", use_container_width=True):
                # Keep pdf_pages but clear quiz state for regeneration
                st.session_state.confirm_regenerate = False
                st.session_state.regenerate_from_pdf = True
                # Clear quiz-related state but keep PDF data
                for key in ["quiz_active", "quiz_state", "all_questions",
                           "questions_by_difficulty", "current_session_id", "report_text"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_regenerate = False
                st.rerun()
        return

    # Action buttons
    if is_pdf_mode:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 View Dashboard", use_container_width=True):
                _clear_quiz_state()
                st.rerun()
        with col2:
            if st.button("🔄 Retry Same Questions", use_container_width=True):
                # Reset quiz state but keep questions
                st.session_state.quiz_state = {
                    "current_difficulty": 4,
                    "asked": set(),
                    "answers": [],
                    "quiz_end": False,
                    "current_q_idx": None,
                    "current_q": None,
                    "show_explanation": False,
                    "last_correct": None,
                }
                if "report_text" in st.session_state:
                    del st.session_state["report_text"]
                # Create new session for retry
                st.session_state.current_session_id = save_quiz_session(
                    st.session_state.user["id"],
                    st.session_state.get("pdf_name", "Retry"),
                    st.session_state.all_questions,
                    0, 0, False
                )
                st.rerun()
        with col3:
            if st.button("📄 Generate New Questions", use_container_width=True):
                st.session_state.confirm_regenerate = True
                st.rerun()

        questions_json = json.dumps(st.session_state.all_questions, indent=2)
        filename = f"ascendquiz_questions_{datetime.now().strftime('%Y-%m-%d')}.json"
        st.download_button(
            label="⬇️ Download Question Pool (JSON)",
            data=questions_json,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
        )
    else:
        # Demo mode - simpler options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View Dashboard", use_container_width=True):
                _clear_quiz_state()
                st.rerun()
        with col2:
            if st.button("🔄 Take Another Quiz", use_container_width=True):
                _clear_quiz_state()
                st.rerun()


def _clear_quiz_state():
    """Helper to clear all quiz-related session state."""
    keys_to_clear = [
        "quiz_active", "quiz_state", "quiz_mode", "all_questions",
        "questions_by_difficulty", "current_session_id", "pdf_pages",
        "pdf_name", "report_text"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# ============== MAIN ==============

def main():
    if not st.session_state.get("logged_in", False):
        render_login_page()
        return
    
    page = render_sidebar()
    
    if page == "📝 Take Quiz":
        render_quiz()
    elif page == "📊 Dashboard":
        render_dashboard()
    elif page == "📜 History":
        render_history()

if __name__ == "__main__":
    main()