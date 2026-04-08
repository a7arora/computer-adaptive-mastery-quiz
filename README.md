# AscendQuiz: An AI-Powered Adaptive Mastery Quiz 🚀

Ascend Quiz is a fully functional Streamlit web app that automatically generates personalized, mastery-driven quizzes from any uploaded PDF (lecture notes, textbook excerpts, etc). Built with Deepseek's R1-0528 API, the app uses generative AI to create high-quality, curriculum-agnostic assessments with adaptive difficulty and real-time feedback.

Main file: [ascendquiz_db_v3.py](https://github.com/tomragus/computer-adaptive-mastery-quiz/blob/main/ascendquiz_db.py)

### To run the quiz from your device:
Download the streamlit library and then set up your "secrets.toml" file (in your directory create a hidden ".stramlit" folder and put into it a "secrets.toml" file containing your Gemini API key: type GEMINI_API_KEY = "key". To get your Gemini API key along with $300 worth of free API credits, create an account on Google AI Studio and go to [https://aistudio.google.com/api-keys](https://aistudio.google.com/api-keys) to get your key).

Once your environment is set up, run using 'streamlit run ascendquiz_db_v3.py'

#### What's new:
- User authentication — Create account, log back in to see progress
- Topic tagging — Each question is tagged with a topic (e.g., "Cell Biology", "Algorithms"), enabling per-topic performance tracking
- Dashboard — View quizzes taken, average score, and potential weaker topics
- Quiz history — Review all past quiz attempts with scores and mastery status

### This project is a collaboration between Ashley Zhou, Tom Ragus, Ethan Baquarian, and Timothy Chen. It is a work in progress and we are actively working to improve it.

A key challenges (as of April 2026):
- Having a determined set of quiz questions generated in the question pool at the start of the quiz (current problem: not enough questions are generated, making the quiz sometimes end early due to lack of questions, even if the user answers them correctly). IDEA: running multiple API calls in parallel across difficulty distribution (instead of calling once to generate 12-10-6-2, call twice to generate 6-5-3-1: asking for fewer question per API call should increase reliability)
- Adding additional features, reworking the "mastery" feature, updating dashboard and history features

<img width="1391" height="786" alt="Screenshot 2026-04-07 at 5 37 55 PM" src="https://github.com/user-attachments/assets/834595f9-7ec2-4c6e-a9f8-1712e72a2a64" />

### Update: as of Apr 7, 2026
I decided to overhaul the architecture a bit to simplify the app. In its current configuration I felt it was becoming overly complex and I figured it would be best to optimize the core quizzing functionality before diving into extra features.

The previous version has been saved as "ascendqiz_db_v2.py", and the simplified version has been saved as "ascendquiz_db_v3.py". This newer version has been stripped of some extra features, including the dashboard, history, and demo quiz - I figured this would help us in staying focused on addressing some of the fundamental technical issues. This new version has a fixed 20-question pool, and it doesn't rely on the large prompt call like the previous version did to build out the quiz. Instead, it makes 4 API calls in parallel, one for each difficulty distribution (user selects easy/medium/hard). We still cannot guarantee enough questions will be generated, but it attempts to generate 30 total and proceeds as long as it reaches 20, making 2 attempts before proceeding. This configuration generates enough questions MOST of the time, but still fails occasionally.

I urge my collaborators to try out this new version in the days leading up to our first team meeting, at which point we can decide whether to build on the simplified version or stick to improving the full version. Since we are starting a new quarter with a new team, I personally think it will be easier to start building on top of this simpler version of the app.
