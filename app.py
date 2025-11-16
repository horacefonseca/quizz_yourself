"""
Interactive Quiz Application
Built with Streamlit - Works with any YAML question bank
"""

import streamlit as st
import os
from pathlib import Path
from utils import (
    load_yaml_questions,
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
            ["Upload YAML file", "Use bundled quiz"]
        )

        questions_loaded = False

        if source_option == "Upload YAML file":
            uploaded_file = st.file_uploader(
                "Upload your YAML question bank",
                type=['yaml', 'yml'],
                help="Upload a YAML file containing quiz questions"
            )

            if uploaded_file is not None:
                file_content = uploaded_file.read().decode('utf-8')
                questions, error = load_yaml_questions(file_content)

                if error:
                    st.error(f"Error loading questions: {error}")
                else:
                    st.session_state.questions = questions
                    questions_loaded = True
                    st.success(f"✅ Loaded {len(questions)} questions!")

        else:  # Use bundled quiz
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

                        questions, error = load_yaml_questions(file_content)

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
