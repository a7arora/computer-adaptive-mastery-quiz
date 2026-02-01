import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random

# Load Gemini API key from Streamlit secrets
API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
MODEL_NAME = "gemini-2.5-pro"

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def get_chunks_by_token(pages):
    """
    Chunks the extracted PDF text based on a 10,000 token limit per chunk.

    - If total tokens <= 10k, it returns one chunk.
    - If total tokens <= 20k, it returns two chunks.
    - If total tokens > 20k, it randomly selects two chunks.
    """
    # Combine all page text into a single string
    full_text = "\n\n".join(pages)

    # Approximate token count, assuming 1 token is about 4 characters
    TOKEN_CHUNK_SIZE = 10000 * 4
    
    # Create all possible chunks from the full text
    all_text_chunks = [full_text[i:i + TOKEN_CHUNK_SIZE] for i in range(0, len(full_text), TOKEN_CHUNK_SIZE)]
    
    num_chunks = len(all_text_chunks)

    if num_chunks <= 2:
        # If the text results in one or two chunks (<= 100k tokens), use them all.
        return all_text_chunks
    else:
        # If there are more than two chunks (> 100k tokens), pick two at random.
        return random.sample(all_text_chunks, 2)


def generate_prompt(text_chunk):
    return f"""
You are a teacher who is designing a test with multiple choice questions (each with 4 answer choices) to test content from a passage.
Given the following passage or notes, generate exactly 20 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 20 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage.
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
Generate exactly 20 questions that vary across difficulty levels. Questions should test **conceptual understanding and application**, not just recall of text. Use the uploaded material to determine:
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
**Examples of BAD questions (too easy to guess without knowledge):**
❌ **Math**: "What is the derivative of x²?"
   - Options: A) 2x, B) Purple, C) √x, D) The number 7
   - Problem: Option B and D are absurd; anyone can eliminate them
❌ **Biology**: "What does the mitochondrion do?"
   - Options: A) Produces energy, B) Makes the cell blue, C) Stores memories, D) Nothing
   - Problem: "Mitochondrion" sounds like "might" + "power" in English; B/C/D are nonsense
❌ **Physics**: "What happens when you compress a gas at constant temperature?"
   - Options: A) Pressure increases, B) It becomes solid immediately, C) Gravity reverses, D) Mass decreases
   - Problem: B/C/D violate basic logic; correct answer follows from common sense about squeezing things
❌ **History**: "What was the primary cause of the Civil War?"
   - Options: A) Slavery and states' rights (detailed), B) Economic factors, C) Politics, D) War
   - Problem: Option A is much more specific; C and D are too vague/circular
❌ **Computer Science**: Question: What is the primary function of an 'exchange argument' in a proof of correctness for a greedy algorithm? A. To prove that the greedy algorithm is more efficient than a brute-force approach. B. To demonstrate that any optimal solution can be transformed into another optimal solution that includes the first greedy choice. C. To show that every non-greedy choice made at any step will inevitably lead to a suboptimal final solution. D. To calculate the performance bound of the greedy algorithm compared to the optimal solution.
- Problem: When someone sees the term exchange argument, they would believe that it aligns with this answer choice "To demonstrate that any optimal solution can be transformed into another optimal solution that includes the first greedy choice." because transforming one solution into another is exchanging one solution into another, so someone who is clever enough can infer the correct answer from the choices without actual knowledge of the concepts covered in the course just from the name.

**Examples of GOOD questions (require actual domain knowledge):**
✅ **Math**: "For the function f(x) = x² - 4x + 4, what is the nature of its roots?"
   - A) Two distinct real roots
   - B) One repeated real root
   - C) Two complex conjugate roots
   - D) No roots exist
   - Why good: Requires discriminant calculation; all options are mathematically meaningful
✅ **Biology**: "During aerobic respiration, where does the electron transport chain occur?"
   - A) Inner mitochondrial membrane
   - B) Outer mitochondrial membrane
   - C) Mitochondrial matrix
   - D) Cristae folds exclusively
   - Why good: All are parts of mitochondria; requires specific knowledge; A and D both seem correct without deep understanding
✅ **Physics**: "A gas undergoes isothermal compression. Which statement is correct?"
   - A) Internal energy remains constant; work is done on the system
   - B) Internal energy decreases; work is done by the system
   - C) Internal energy increases; no heat is exchanged
   - D) Internal energy remains constant; no work is exchanged
   - Why good: All involve thermodynamic concepts; requires understanding isothermal process properties
✅ **Chemistry**: "Which factor does NOT affect the rate of an enzyme-catalyzed reaction at optimal conditions?"
   - A) Total amount of product already formed
   - B) Substrate concentration
   - C) Enzyme concentration
   - D) Presence of competitive inhibitors
   - Why good: Three factors DO affect rate (plausible); requires understanding enzyme kinetics, not guessing
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
Example: “What is the SI unit of force?” (If the passage explicitly defines it.)
Reasoning check: Any student who read the passage carefully and understood it should get this correct. There should be no trickiness or need for synthesis.
70–84% correct (Easy / Understanding / Single-Step Reasoning)
Requires one step of reasoning or minor inference beyond direct recall.
Students must connect a concept in the passage to a similar context or slightly different phrasing, but it is still straightforward.
Wrong answers should reflect adjacent or related concepts that a partial understanding might confuse.
Example: Distinguishing between two related concepts explained in the passage, or solving a quadratic equation using a method shown in an example, but with different numbers.
Reasoning check: Students need to understand the concept, but the mental load is low; a moderately attentive student can reason it out from the passage.
50–69% correct (Medium / Application / Multi-Step Reasoning)
Requires applying principles from the passage to a new scenario or combining multiple pieces of information.
The passage does not explicitly solve this problem, so students must adapt knowledge.
Distractors should be plausible errors that someone might make if they misapplied formulas, misremembered conditions, or partially understood the concept.
Example: Using a formula from a passage in a context slightly different from the examples given, requiring intermediate calculations or logical steps.
Reasoning check: Students must integrate knowledge, not just recall. Simple elimination of absurd answers is not sufficient.
30–49% correct (Hard / Analysis / Synthesis)
Requires deep understanding, integration, or analysis of multiple concepts in the passage.
Students must infer relationships, compare methods, or predict outcomes not directly explained.
Wrong answers should seem correct to someone with partial understanding, exploiting subtle distinctions or counterintuitive results.
Example: Predicting how two interacting variables affect an outcome based on multiple sections of the passage.
Reasoning check: Requires careful thinking and cannot be answered by rote memorization or simple logic alone.
Below 30% correct (Very Hard / Evaluation or Creation)
Requires expert-level judgment, design, or synthesis, combining multiple principles in novel ways.
Multiple answers might seem defensible; students must evaluate, critique, or generate solutions based on passage principles.
Distractors reflect plausible alternative interpretations, partial understanding, or common advanced mistakes.
Example: Designing an experiment, predicting outcomes in a complex system, or choosing between competing strategies using principles from the passage.
Reasoning check: Even strong students may struggle; requires higher-order thinking and creativity, not just reasoning from examples

You have a tendency to think questions are harder than they really are for students and think that the distractors are much better than they really are, so please keep that in mind when generating questions, especially for medium-hard and hard questions. Really try to make the medium-hard and hard questions require deep mastery of the material beyond what is expected of students who typically study material at this level and not just be simple applications of the material. Really try to make the distractors good enough so that students would not even guess on the question if there was negative marking schemes in exams like the older SAT or AMC 10. 
Please do your best to make the hardest questions truly super difficult, even for students who are well prepared and at top schools/universities, while also making sure that they are correct
**IMPORTANT DIFFICULTY CHECK:**
Before assigning estimated_correct_pct below 70%, ask yourself:
1. Could a clever person with no domain knowledge eliminate 2+ options using logic alone?
2. Does the question wording hint at the answer through word associations?
3. Are any options absurd enough that anyone would eliminate them?
4. Could someone pattern-match (longest/most specific option is often correct)?
5. Is the question something for which the correct answer is directly in the reading?
If you answered YES to any of these, the question is easier than you think. Increase the percentage OR redesign the options.
**Requirements**:
- 5 easy (≥85%), 5 medium (60–84%), 5 medium-hard (40-60%), 5 hard (<40%)
**Each question must include the following fields:**
- "question": A clear, concise, and unambiguous question that tests understanding of concepts from the passage. The question should be COMPLETELY SELF-CONTAINED with all necessary context included. Never reference "the passage," "the text," specific examples by position (first, second, etc.), or figures/tables. Ask about the concept directly.
- - "options": An array of exactly 4 strings in this exact format:
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
Return **only** a valid JSON array of 20 questions.  Focus on testing conceptual understanding rather than text memorization. 
Do not include any text, commentary, or markdown fences. 
Output must begin with `[` and end with `]` — no explanations outside JSON.
Passage:
{text_chunk}
"""

def call_gemini_api(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 13000
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
    """
    Extracts the JSON part from a model response.
    Strips ```json fences, trailing commentary, and truncates at the last bracket.
    """
    text = text.strip()
    # Remove ```json ... ``` fences (more flexible pattern)
    fence_patterns = [
        r"```json\s*(.*?)```",  # ```json content ```
        r"```\s*(.*?)```",  # ``` content ```
        r"`{3,}\s*json\s*(.*?)`{3,}",  # Multiple backticks with json
        r"`{3,}\s*(.*?)`{3,}"  # Multiple backticks without json
    ]
    
    for pattern in fence_patterns:
        fence_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
            break
    
    # Find the JSON array boundaries
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
        return text.strip()
    
    # Fallback: look for object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
        return text.strip()
    
    return text

def repair_json(text: str) -> str:
    """
    Repairs model-generated JSON output that may be truncated or malformed.
    Ensures valid array syntax and removes dangling commas.
    """
    text = re.sub(r'```(?:json)?', '', text)
    text = text.replace('```', '').strip()

    # Keep only the JSON-like section
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        text = text[start:end + 1]
    else:
        # fallback for single object JSON
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = f"[{text[start:end + 1]}]"

    # --- Critical fix: ensure clean object separation ---
    # 1. Remove incomplete trailing objects (missing closing brace)
    open_braces = text.count('{')
    close_braces = text.count('}')
    if close_braces < open_braces:
        # trim off the incomplete last object
        last_full = text.rfind('}')
        if last_full != -1:
            text = text[:last_full + 1]
        text += "]" if not text.endswith(']') else ""

    # 2. Fix any "}, {" merges without comma or bracket
    text = re.sub(r'}\s*{', '}, {', text)

    # 3. Remove trailing commas before closing brackets
    text = re.sub(r',\s*([\]}])', r'\1', text)

    # 4. Ensure the array starts and ends properly
    if not text.startswith('['):
        text = '[' + text
    if not text.endswith(']'):
        text = text + ']'

    # Final clean trim
    return text.strip()



def parse_question_json(text: str):
    """
    Parse JSON with better error handling and debugging
    """
    print(f"Raw API response length: {len(text)}")
    print(f"Raw API response (first 200 chars): {text[:200]}")
    
    cleaned = clean_response_text(text)
    print(f"Cleaned text length: {len(cleaned)}")
    print(f"Cleaned text (first 200 chars): {cleaned[:200]}")
    
    cleaned = repair_json(cleaned)
    print(f"Repaired text (first 200 chars): {cleaned[:200]}")
    
    # Try standard JSON parsing first
    try:
        result = json.loads(cleaned)
        print(f"Successfully parsed {len(result) if isinstance(result, list) else 1} questions")
        return result
    except json.JSONDecodeError as e:
        print(f"Standard JSON parsing failed: {e}")
        
        # Try json5 as fallback
        try:
            import json5
            result = json5.loads(cleaned)
            print(f"JSON5 parsing successful: {len(result) if isinstance(result, list) else 1} questions")
            return result
        except Exception as e2:
            print(f"JSON5 parsing also failed: {e2}")
            
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
                    print(f"Manual extraction successful: {len(questions)} questions")
                    return questions
            except:
                pass
            
            st.error("⚠️ JSON parse failed:")
            st.error(f"Standard JSON error: {e}")
            st.error(f"JSON5 error: {e2}")
            st.text("Raw cleaned text (first 1000 chars):")
            st.text(cleaned[:1000])
            
            return []

def filter_invalid_difficulty_alignment(questions):
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

def find_next_difficulty(current_diff, going_up, asked, all_qs):
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
    available = pick_question(current_diff, asked, all_qs)
    if not available:
        return current_diff, None, None
    idx, q = random.choice(available)
    return current_diff, idx, q

def accuracy_on_levels(answers, levels):
    filtered = [c for d, c in answers if d in levels]
    return sum(filtered) / len(filtered) if filtered else 0

def compute_mastery_score(answers):
    mastery_bands = {
        (1, 2): 25,
        (3, 4): 65,
        (5, 6): 85,
        (7, 8): 100
    }
    min_attempts_required = 3
    band_scores = []
    for levels, weight in mastery_bands.items():
        relevant = [correct for d, correct in answers if d in levels]
        attempts = len(relevant)
        if attempts == 0:
            continue
        acc = sum(relevant) / attempts
        normalized_score = max((acc - 0.25) / 0.75, 0)
        if attempts < min_attempts_required:
            scaled_score = normalized_score * weight * (attempts / min_attempts_required)
            band_scores.append(scaled_score)
        else:
            band_score = normalized_score * weight
            band_scores.append(band_score)
    if not band_scores:
        return 0
    return int(round(max(band_scores)))

# Adding a banner title with rocket emoji
st.markdown('<h1 class="main-title">🚀 AscendQuiz</h1>', unsafe_allow_html=True)

# Styling block that locks the answer selection after submitting an answer
st.markdown("""
    <style>
        /* Prevent the labels of disabled radio buttons from dimming */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label [data-testid="stWidgetLabel"] p {
            opacity: 1 !important;
            color: inherit !important;
        }
        
        /* Optional: Keep the radio options themselves clear while disabled */
        div[data-testid="stRadio"] label {
            opacity: 1 !important;
        }

        /* Keep the selected option text bold even when locked */
        div[data-testid="stRadio"] input[checked] + div p {
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Styling block for overall UI, slight background gradient
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
            color: #1f1f1f; /* Explicitly set default text to dark grey */
        }
        
        /* Ensure all headers, markdown, and labels are visible */
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label {
            color: #1f1f1f !important;
        }

        .main-title {
            color: #2e4a7d !important; /* Keep your specific blue for the title */
            font-size: 3rem !important;
            font-weight: 800;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        /* Fix for radio buttons and other widgets */
        div[data-testid="stRadio"] label p {
            color: #1f1f1f !important;
        }
    </style>
""", unsafe_allow_html=True)

def render_mastery_bar(score):
    if score < 30:
        color = "#dc3545"  # red
        text_color = "white"
    elif score < 70:
        color = "#ffc107"  # yellow
        text_color = "black"
    else:
        color = "#28a745"  # green
        text_color = "white"
    st.markdown(f"""
    <style>
        .stApp {{
            padding-top: 70px;
        }}
        .mastery-bar-wrapper {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            width: 100%;
            z-index: 999999;
            background-color: white;
            padding: 12px 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            border-bottom: 1px solid #ddd;
        }}
        .mastery-bar {{
            border: 2px solid #ccc;
            border-radius: 8px;
            overflow: hidden;
            height: 28px;
            width: 100%;
            background-color: #f0f0f0;
            position: relative;
        }}
        .mastery-bar-fill {{
            height: 100%;
            width: {score}%;
            background-color: {color};
            transition: width 0.3s ease;
            position: absolute;
            top: 0;
            left: 0;
        }}
        .mastery-bar-text {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: {text_color if score > 10 else 'black'};
            font-weight: bold;
            font-size: 14px;
            z-index: 2;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
    </style>
    <div class="mastery-bar-wrapper">
        <div style="font-size: 12px; margin-bottom: 4px; color: #666; font-weight: 500;">
            Mastery Progress
        </div>
        <div class="mastery-bar">
            <div class="mastery-bar-fill"></div>
            <div class="mastery-bar-text">{score}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if "all_questions" not in st.session_state:
    st.markdown("""
Welcome to your personalized learning assistant — an AI-powered tool that transforms any PDF into a mastery-based, computer-adaptive quiz.
**How it works:**
This app uses a large language model (LLM) and an adaptive difficulty engine to create multiple-choice questions from your uploaded notes or textbook excerpts. These questions are labeled with how likely students are to answer them correctly, allowing precise control over quiz difficulty.
The quiz adapts in real-time based on your performance. Starting at a medium level, each correct answer raises the difficulty, while incorrect answers lower it — just like the GRE or ALEKS. Once your **mastery score reaches 70% or higher** (calculated using your accuracy weighted by difficulty level), the system considers you to have achieved **mastery** and ends the quiz.
Each question includes:
- Four answer options
- The correct answer
- An explanation
- A predicted correctness percentage

Unlike static tools like Khanmigo, this app uses generative AI to dynamically create the quiz from **your own content** — no rigid question banks required.
**Built using the Gemini 2.5 Pro model**, this app is a proof-of-concept showing what modern AI can do for personalized education. It blends mastery learning, real-time feedback, and adaptive testing into one clean experience. Please keep in mind that it currently takes about 1-3 minutes to generate questions from a pdf... please be patient as it generates questions. Furthermore, it only accepts text output and cannot read handwriting or drawings at this time.
""")
    uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
    if uploaded_pdf:
        with st.spinner("Generating questions..."):
            pages = extract_text_from_pdf(uploaded_pdf)
            
            # This is the new token-based chunking logic
            chunks_to_use = get_chunks_by_token(pages)

            all_questions = []
            for chunk in chunks_to_use:
                if not chunk.strip(): # Skip empty chunks
                    continue
                prompt = generate_prompt(chunk)
                response_text, error = call_gemini_api(prompt)
                if error:
                    st.error("API error: " + error)
                    continue
                parsed = parse_question_json(response_text)
                valid, invalid = filter_invalid_difficulty_alignment(parsed)
                all_questions.extend(valid)
                if "filtered_questions" not in st.session_state:
                    st.session_state.filtered_questions = []
                st.session_state.filtered_questions.extend(invalid)

            if all_questions:
                st.session_state.all_questions = all_questions
                st.session_state.questions_by_difficulty = group_by_difficulty(all_questions)
                st.session_state.quiz_state = {
                    "current_difficulty": 4,
                    "asked": set(),
                    "answers": [],
                    "quiz_end": False,
                    "current_q_idx": None,
                    "current_q": None,
                    "show_explanation": False,
                    "last_correct": None,
                    "last_explanation": None,
                }
                st.success("✅ Questions generated! Starting the quiz...")
                st.session_state.quiz_ready = True
                st.rerun()
            else:
                st.error("No questions were generated.")
elif "quiz_ready" in st.session_state and st.session_state.quiz_ready:
    all_qs = st.session_state.questions_by_difficulty
    state = st.session_state.get("quiz_state", None)
    if state is None:
        st.warning("Quiz state not found. Please restart the app or re-upload a PDF.")
        st.stop()
    score = compute_mastery_score(state.get("answers", []))
    render_mastery_bar(score)
    if not state["quiz_end"]:
        if state["current_q"] is None and not state.get("show_explanation", False):
            diff, idx, q = get_next_question(state["current_difficulty"], state["asked"], all_qs)
            if q is None:
                state["quiz_end"] = True
            else:
                state["current_q"] = q
                state["current_q_idx"] = idx
                state["current_difficulty"] = diff
    if not state["quiz_end"] and state["current_q"]:
        q = state["current_q"]
        idx = state["current_q_idx"]
        st.markdown(f"### Question (Difficulty {state['current_difficulty']})")
        st.markdown(q["question"], unsafe_allow_html=True)
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
        selected = st.radio("Select your answer:", options=rendered_options, key=f"radio_{idx}", index=None, disabled=state.get("show_explanation", False))
        if st.button("Submit Answer", key=f"submit_{idx}", disabled=state.get("show_explanation", False)):
            if selected is None:
                st.warning("Please select an answer before submitting.")
            else:
                selected_letter = selected.split(".")[0].strip().upper()
                letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
                correct_letter = q["correct_answer"].strip().upper()
                correct_index = letter_to_index.get(correct_letter, None)
                if correct_index is None:
                    st.error("⚠️ Question error: Correct answer letter invalid.")
                    state["quiz_end"] = True
                    st.stop()
                correct = (selected_letter == correct_letter)
                state["asked"].add((state["current_difficulty"], idx))
                state["answers"].append((state["current_difficulty"], correct))
                state["last_correct"] = correct
                state["last_explanation"] = q["explanation"]
                state["show_explanation"] = True
                score = compute_mastery_score(state["answers"])
                if score >= 70:
                    state["quiz_end"] = True
                st.rerun()
        if state.get("show_explanation", False):
            if state["last_correct"]:
                st.success("✅ Correct!")
                st.markdown(state["last_explanation"], unsafe_allow_html=True)
            else:
                st.markdown("❌ **Incorrect.**", unsafe_allow_html=True)
                st.markdown(state["last_explanation"], unsafe_allow_html=True)
            if st.button("Next Question"):
                def find_next_difficulty(current_diff, going_up, asked, all_qs):
                    diffs = range(current_diff + 1, 9) if going_up else range(current_diff - 1, 0, -1)
                    for d in diffs:
                        if pick_question(d, asked, all_qs):
                            return d
                    return current_diff
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
                state["last_explanation"] = None
                st.rerun()
    elif state["quiz_end"]:
        acc = accuracy_on_levels(state["answers"], [5, 6, 7, 8])
        hard_attempts = len([1 for d, _ in state["answers"] if d >= 5])
        st.markdown("## Quiz Completed 🎉")
        if score >= 70:
            st.success(f"🎉 You have mastered the content! Your mastery score is {score}%. Great job!")
        else:
            st.warning(f"Mastery not yet achieved. Your mastery score is {score}%. Review the material and try again.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Retry Quiz", use_container_width=True):
                # Reset only the quiz progress state while keeping the generated questions
                st.session_state.quiz_state = {
                    "current_difficulty": 4,
                    "asked": set(),
                    "answers": [],
                    "quiz_end": False,
                    "current_q_idx": None,
                    "current_q": None,
                    "show_explanation": False,
                    "last_correct": None,
                    "last_explanation": None,
                }
                st.rerun()

        with col2:
            if st.button("🏠 Home", use_container_width=True):
                # Clear all session data to return to the PDF upload screen
                keys_to_clear = [
                    "all_questions", 
                    "questions_by_difficulty", 
                    "quiz_state", 
                    "quiz_ready", 
                    "filtered_questions"
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        if "all_questions" in st.session_state:
            all_qs_json = json.dumps(st.session_state.all_questions, indent=2)
            st.download_button(
                label="📥 Download All Quiz Questions (JSON)",
                data=all_qs_json,
                file_name="ascendquiz_questions.json",
                mime="application/json"
            )


