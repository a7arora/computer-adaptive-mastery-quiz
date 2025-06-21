# computer-adaptive-mastery-quiz
This project is a fully functional Streamlit web app that generates and delivers computer-adaptive, mastery-based quizzes from any uploaded PDF document. Using a large language model (LLM) and a dynamic difficulty engine, it personalizes assessment in real-time — adjusting questions based on student performance and estimating how many students would likely get each question correct.

Built independently using the meta-llama/llama-4-scout-17b-16e-instruct model from Groq, the app turns uploaded educational content (like lecture notes or textbook excerpts) into 15 multiple-choice questions that vary in difficulty. The questions are automatically labeled with an estimated percent of students who would get them right, allowing for fine-grained control over quiz difficulty and personalization.

The heart of this app lies in its adaptive testing loop. The quiz starts at a medium difficulty level (questions expected to be answered correctly by ~70% of students). If the user answers correctly, the next question is made harder. If they answer incorrectly, it becomes easier. This closely mirrors the behavior of standardized computer-adaptive testing systems used in platforms like ALEKS and the GRE — but with real-time generation from any content.

Each question includes four options, a correct answer, an explanation, and a predicted correctness percentage. The app then groups these into 8 difficulty tiers (e.g., "very easy" to "very hard") based on the estimated correctness rates. This enables nuanced control and selection of follow-up questions that are neither too easy nor too frustrating, helping to maintain engagement while promoting learning.

A key educational innovation in this app is its mastery tracking system. A student is considered to have “achieved mastery” when they answer at least five hard questions (difficulty level 6 or higher) with a minimum accuracy of 75%. The quiz then automatically ends and provides feedback on the student’s performance — creating an outcome-based assessment experience rather than one based on completion or fixed length.

Unlike traditional quizzes or most AI-assisted quiz generators (such as Khanmigo), this tool does not rely on predefined curricula or rigid question banks. Instead, it uses generative AI to create quiz content dynamically based on user-uploaded PDFs — allowing maximum flexibility for educators across subjects and grade levels.

The backend includes a call_groq_api function that queries the LLM with a structured prompt to return JSON-formatted question objects. These are parsed, difficulty-ranked, and fed into the adaptive engine. The frontend, built entirely in Streamlit, manages session state, user interaction, and quiz progression through a clean, minimal interface.

Another important feature is immediate feedback. After each submission, users are told whether they were correct and are shown the rationale behind the correct answer. This real-time feedback loop encourages reflective learning and aligns with the pedagogical principles of active recall and spaced repetition.

Technically, the app supports robust error handling, intelligent fallback if questions of a certain difficulty are exhausted, and even smart difficulty search (searching nearby levels if the exact difficulty isn’t available). These engineering details help maintain a seamless user experience even in edge cases.

In sum, this app represents a working proof-of-concept for what modern AI can bring to personalized education. It combines mastery learning, computer-adaptive testing, and generative AI into a single, interactive experience. It goes beyond existing educational tools by adapting not just content delivery, but content generation, difficulty estimation, and instructional feedback — all in real time.
