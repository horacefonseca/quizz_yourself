"""
Interactive Quiz Application
Built with Streamlit - Works with any YAML question bank
"""

import streamlit as st
import os
from pathlib import Path
from utils import (
    load_questions,
    sample_questions,
    grade_quiz,
    generate_pdf,
    calculate_quiz_length
)

# Page configuration
st.set_page_config(
    page_title="Interactive Quiz App",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'grading_results' not in st.session_state:
    st.session_state.grading_results = None


def reset_quiz():
    """Reset quiz to initial state"""
    st.session_state.quiz_started = False
    st.session_state.quiz_questions = []
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.grading_results = None


def main():
    """Main application"""

    # Title
    st.title("📝 Interactive Quiz Application")
    st.markdown("---")

    # Sidebar for quiz setup
    with st.sidebar:
        st.header("Quiz Setup")

        # Step 1: Question Bank Selection
        st.subheader("1. Select Question Bank")

        source_option = st.radio(
            "Choose question source:",
            ["Upload YAML/Markdown file", "Paste ChatGPT-formatted text", "Use bundled quiz"]
        )

    # Main area content based on selection
    questions_loaded = False

    if source_option == "Paste ChatGPT-formatted text":
        # CHATGPT WORKFLOW IN CENTER (moved from sidebar)
        st.header("📝 Create Quiz from Raw Text with ChatGPT AI")
        st.info("💡 **Quick Steps:** 1️⃣ Copy instructions → 2️⃣ Open ChatGPT → 3️⃣ Paste & add your text → 4️⃣ Copy result back here")

        # ChatGPT instructions
        gemini_instructions = """Please convert the following raw text into a structured quiz format and coding text format ready to cut and paste with indentations. Follow these rules EXACTLY:

FORMAT RULES:
Each question must start with "QUESTION" followed by a number (QUESTION 1, QUESTION 2, etc.)
Each question must have these fields (one per line):
Type: (either "mc" for multiple choice or "open" for open-ended)
Question: (the question text)

FOR MULTIPLE CHOICE (mc):
A) (first option)
B) (second option)
C) (third option)
D) (fourth option)
Correct: (the correct letter: A, B, C, or D)

FOR OPEN-ENDED (open):
Answer: (the correct answer text)

OPTIONAL FIELDS (for both types):
Explanation: (5-15 words explaining why the answer is correct)
Chapter: (chapter or topic reference)

Leave a blank line between questions
Use EXACTLY this format - no extra formatting, no bold, no italics

EXAMPLE OUTPUT:

QUESTION 1
Type: mc
Question: What is the range of valid probability values?
A) Between -1 and 1
B) Between 0 and 1
C) Between 0 and 100
D) Any positive number
Correct: B
Explanation: Probabilities must be between 0 (impossible) and 1 (certain).
Chapter: 1.1

QUESTION 2
Type: open
Question: Define binary classification.
Answer: Classification with exactly two possible class labels
Explanation: Binary classification has exactly two possible outcomes.
Chapter: 3

====================================================
[Paste your raw text here]"""

        # Action buttons row
        col1, col2 = st.columns([1, 1])

        with col1:
            # Copy button with JavaScript
            copy_button_html = f"""
            <button onclick="copyToClipboard()" style="
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                width: 100%;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: all 0.3s;
            " onmouseover="this.style.backgroundColor='#45a049'"
               onmouseout="this.style.backgroundColor='#4CAF50'">
                📋 Copy Instructions
            </button>
            <textarea id="gemini-instructions" style="position: absolute; left: -9999px;">{gemini_instructions}</textarea>
            <p id="copy-status" style="color: green; font-weight: bold; margin-top: 8px; min-height: 24px;"></p>
            <script>
            function copyToClipboard() {{
                var copyText = document.getElementById("gemini-instructions");
                copyText.style.position = "static";
                copyText.select();
                copyText.setSelectionRange(0, 99999);
                document.execCommand("copy");
                copyText.style.position = "absolute";
                document.getElementById("copy-status").innerHTML = "✅ Copied to clipboard!";
                setTimeout(function() {{
                    document.getElementById("copy-status").innerHTML = "";
                }}, 3000);
            }}
            </script>
            """
            st.markdown(copy_button_html, unsafe_allow_html=True)

        with col2:
            # ChatGPT link button
            chatgpt_button_html = """
            <a href="https://chatgpt.com" target="_blank" style="text-decoration: none;">
                <button style="
                    background: linear-gradient(135deg, #10a37f 0%, #1a7f64 100%);
                    color: white;
                    padding: 12px 24px;
                    font-size: 16px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    width: 100%;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    transition: all 0.3s;
                " onmouseover="this.style.transform='scale(1.05)'"
                   onmouseout="this.style.transform='scale(1)'">
                    🤖 Open ChatGPT
                </button>
            </a>
            """
            st.markdown(chatgpt_button_html, unsafe_allow_html=True)

        # Expandable view of instructions (optional, for reference)
        with st.expander("📖 View Full Instructions"):
            st.code(gemini_instructions, language=None)

        st.markdown("---")

        # Text area for pasting ChatGPT output (20,000 words ≈ 120,000 characters)
        chatgpt_text = st.text_area(
            "Paste ChatGPT's formatted output here:",
            height=300,
            max_chars=120000,
            placeholder="Paste the formatted questions from ChatGPT here...\n\nExample:\nQUESTION 1\nType: mc\nQuestion: What is...?\nA) Option A\nB) Option B\n..."
        )

        if chatgpt_text and chatgpt_text.strip():
            # Count approximate words
            word_count = len(chatgpt_text.split())
            st.caption(f"📝 Approximately {word_count:,} words pasted")

            # Parse the ChatGPT-formatted text
            questions, error = load_questions(chatgpt_text, 'gemini')

            if error:
                st.error(f"❌ Error parsing format: {error}")
                st.info("💡 Make sure you copied the EXACT format from ChatGPT. Check the instructions above.")
            else:
                st.session_state.questions = questions
                questions_loaded = True
                st.success(f"✅ Successfully parsed {len(questions)} questions!")

    # Other options stay in sidebar
    with st.sidebar:
        if source_option == "Upload YAML/Markdown file":
            uploaded_file = st.file_uploader(
                "Upload your question bank",
                type=['yaml', 'yml', 'md'],
                help="Upload a YAML or Markdown file containing quiz questions"
            )

            if uploaded_file is not None:
                file_content = uploaded_file.read().decode('utf-8')
                file_extension = uploaded_file.name.split('.')[-1].lower()
                questions, error = load_questions(file_content, file_extension)

                if error:
                    st.error(f"Error loading questions: {error}")
                else:
                    st.session_state.questions = questions
                    questions_loaded = True
                    st.success(f"✅ Loaded {len(questions)} questions!")

        elif source_option == "Use bundled quiz":
            # Look for sample quizzes in sample_quizzes folder
            sample_dir = Path(__file__).parent / "sample_quizzes"

            if sample_dir.exists():
                sample_files = list(sample_dir.glob("*.yaml")) + list(sample_dir.glob("*.yml"))

                if sample_files:
                    selected_file = st.selectbox(
                        "Select a quiz:",
                        sample_files,
                        format_func=lambda x: x.stem
                    )

                    if selected_file:
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            file_content = f.read()

                        file_extension = selected_file.suffix.lstrip('.').lower()
                        questions, error = load_questions(file_content, file_extension)

                        if error:
                            st.error(f"Error loading questions: {error}")
                        else:
                            st.session_state.questions = questions
                            questions_loaded = True
                            st.success(f"✅ Loaded {len(questions)} questions!")
                else:
                    st.warning("No sample quizzes found in sample_quizzes folder")
            else:
                st.warning("sample_quizzes folder not found")

        # Step 2: Quiz Length Selection
        if questions_loaded and st.session_state.questions:
            st.markdown("---")
            st.subheader("2. Quiz Length")

            total_questions = len(st.session_state.questions)
            st.info(f"Total available questions: {total_questions}")

            # Percentage slider
            percentage = st.slider(
                "Select percentage of questions:",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="Choose what percentage of the question bank to include"
            )

            suggested_num = int(total_questions * percentage / 100)
            st.write(f"Suggested: {suggested_num} questions")

            # Custom override
            use_custom = st.checkbox("Use custom number of questions")

            if use_custom:
                custom_num = st.number_input(
                    "Number of questions:",
                    min_value=1,
                    max_value=total_questions,
                    value=min(suggested_num, total_questions)
                )
                final_num = custom_num
            else:
                final_num = suggested_num

            st.markdown("---")

            # Start Quiz Button
            if not st.session_state.quiz_started:
                if st.button("🚀 Start Quiz", use_container_width=True):
                    # Sample questions
                    st.session_state.quiz_questions = sample_questions(
                        st.session_state.questions,
                        final_num
                    )
                    st.session_state.quiz_started = True
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()
            else:
                if st.button("🔄 Reset Quiz", use_container_width=True):
                    reset_quiz()
                    st.rerun()

    # Main content area
    if not st.session_state.quiz_started:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Interactive Quiz Application! 👋

        ### How to use:
        1. **Select a question bank** from the sidebar (upload or use bundled)
        2. **Choose quiz length** using the percentage slider or custom number
        3. **Click "Start Quiz"** to begin
        4. **Answer all questions** and submit
        5. **Download your results** as a PDF

        ### Supported File Formats:
        📄 **YAML** (.yaml, .yml) - Structured format
        📝 **Markdown** (.md) - Human-readable format

        ### Supported Question Types:
        - **Multiple Choice (MC):** Select from provided options
        - **Open Questions:** Type your answer

        ### Features:
        ✅ Random question sampling
        ✅ Instant grading
        ✅ Detailed explanations
        ✅ PDF results export
        """)

    elif not st.session_state.quiz_submitted:
        # Quiz Interface
        st.header(f"Quiz: {len(st.session_state.quiz_questions)} Questions")

        st.markdown("---")

        # Display all questions
        for i, q in enumerate(st.session_state.quiz_questions, 1):
            with st.container():
                st.subheader(f"Question {i} of {len(st.session_state.quiz_questions)}")

                # Question text
                st.markdown(f"**{q['question']}**")

                # Chapter reference if available
                if 'chapter' in q:
                    st.caption(f"📚 Chapter: {q['chapter']}")

                # Answer input based on question type
                if q['type'] == 'mc':
                    # Multiple choice
                    options = q['options']
                    user_answer = st.radio(
                        "Select your answer:",
                        options,
                        key=f"q_{q['id']}",
                        index=None
                    )

                    if user_answer:
                        # Extract letter (A, B, C, etc.)
                        answer_letter = user_answer.split(')')[0].strip()
                        st.session_state.user_answers[q['id']] = answer_letter

                elif q['type'] == 'open':
                    # Open question
                    user_answer = st.text_input(
                        "Your answer:",
                        key=f"q_{q['id']}",
                        placeholder="Type your answer here..."
                    )

                    if user_answer:
                        st.session_state.user_answers[q['id']] = user_answer

                st.markdown("---")

        # Submit button
        st.markdown("### Ready to submit?")

        answered = len(st.session_state.user_answers)
        total = len(st.session_state.quiz_questions)

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.info(f"Questions answered: {answered}/{total}")

        with col2:
            if answered < total:
                st.warning("⚠️ Some questions unanswered")

        with col3:
            if st.button("📤 Submit Quiz", type="primary", use_container_width=True):
                # Grade the quiz
                st.session_state.grading_results = grade_quiz(
                    st.session_state.quiz_questions,
                    st.session_state.user_answers
                )
                st.session_state.quiz_submitted = True
                st.rerun()

    else:
        # Results Screen
        results = st.session_state.grading_results

        st.header("📊 Quiz Results")
        st.markdown("---")

        # Score summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Score",
                f"{results['score_percentage']:.1f}%",
                delta=None
            )

        with col2:
            st.metric(
                "Correct",
                results['correct'],
                delta=None
            )

        with col3:
            st.metric(
                "Incorrect",
                results['incorrect'],
                delta=None
            )

        st.markdown("---")

        # Performance indicator
        score = results['score_percentage']
        if score >= 90:
            st.success("🎉 Excellent! Outstanding performance!")
        elif score >= 75:
            st.success("✅ Great job! Very good performance!")
        elif score >= 60:
            st.info("👍 Good work! Room for improvement.")
        else:
            st.warning("📚 Keep studying! Review the explanations below.")

        st.markdown("---")

        # Detailed results
        st.subheader("Detailed Answer Sheet")

        for i, result in enumerate(results['results'], 1):
            with st.expander(
                f"Question {i}: {'✅ Correct' if result['is_correct'] else '❌ Incorrect'}",
                expanded=not result['is_correct']  # Auto-expand incorrect answers
            ):
                st.markdown(f"**Question:** {result['question']}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Your answer:**")
                    if result['is_correct']:
                        st.success(result['user_answer'])
                    else:
                        st.error(result['user_answer'])

                with col2:
                    st.markdown(f"**Correct answer:**")
                    st.info(result['correct_answer'])

                st.markdown(f"**💡 Explanation:** {result['explanation']}")

                if result['chapter'] != 'N/A':
                    st.caption(f"📚 Chapter: {result['chapter']}")

        st.markdown("---")

        # PDF Download
        st.subheader("📄 Download Results")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("Download a PDF report of your quiz results including all questions, answers, and explanations.")

        with col2:
            # Generate PDF
            pdf_bytes = generate_pdf(results, "Quiz Results")

            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"quiz_results_{st.session_state.grading_results['score_percentage']:.0f}pct.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # Retake button
        st.markdown("---")
        if st.button("🔄 Take Another Quiz", use_container_width=False):
            reset_quiz()
            st.rerun()


if __name__ == "__main__":
    main()
