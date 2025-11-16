"""
Utility functions for the Quiz Application
Handles YAML loading, question sampling, grading, and PDF generation
"""

import yaml
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple
from fpdf import FPDF


def load_yaml_questions(file_content: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load questions from YAML content

    Args:
        file_content: String content of YAML file

    Returns:
        Tuple of (list of questions, error message if any)
    """
    try:
        questions = yaml.safe_load(file_content)

        if not questions:
            return [], "YAML file is empty"

        if not isinstance(questions, list):
            return [], "YAML must contain a list of questions"

        # Validate question structure
        required_fields = ['id', 'question', 'type']
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                return [], f"Question {i+1} is not properly formatted"

            for field in required_fields:
                if field not in q:
                    return [], f"Question {i+1} missing required field: {field}"

            # Validate question type
            if q['type'] not in ['mc', 'open']:
                return [], f"Question {i+1} has invalid type: {q['type']} (must be 'mc' or 'open')"

            # Validate MC questions
            if q['type'] == 'mc':
                if 'options' not in q:
                    return [], f"MC Question {i+1} missing 'options'"
                if 'correct' not in q:
                    return [], f"MC Question {i+1} missing 'correct' answer"

            # Validate open questions
            if q['type'] == 'open':
                if 'answer' not in q:
                    return [], f"Open Question {i+1} missing 'answer'"

        return questions, ""

    except yaml.YAMLError as e:
        return [], f"YAML parsing error: {str(e)}"
    except Exception as e:
        return [], f"Error loading questions: {str(e)}"


def sample_questions(questions: List[Dict[str, Any]], num_questions: int) -> List[Dict[str, Any]]:
    """
    Randomly sample questions without replacement

    Args:
        questions: List of all questions
        num_questions: Number of questions to sample

    Returns:
        List of sampled questions
    """
    if num_questions >= len(questions):
        return questions.copy()

    return random.sample(questions, num_questions)


def grade_quiz(questions: List[Dict[str, Any]], user_answers: Dict[int, str]) -> Dict[str, Any]:
    """
    Grade the quiz by comparing user answers to correct answers

    Args:
        questions: List of quiz questions
        user_answers: Dictionary mapping question IDs to user answers

    Returns:
        Dictionary containing grading results
    """
    total = len(questions)
    correct = 0
    results = []

    for q in questions:
        q_id = q['id']
        user_answer = user_answers.get(q_id, "").strip()

        if q['type'] == 'mc':
            correct_answer = q['correct']
            is_correct = user_answer.upper() == correct_answer.upper()
        else:  # open question
            # For open questions, exact match (case-insensitive)
            correct_answer = q['answer']
            is_correct = user_answer.lower() == correct_answer.lower()

        if is_correct:
            correct += 1

        results.append({
            'id': q_id,
            'question': q['question'],
            'type': q['type'],
            'user_answer': user_answer if user_answer else "No answer",
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'explanation': q.get('explanation', 'No explanation provided'),
            'chapter': q.get('chapter', 'N/A')
        })

    score_percentage = (correct / total * 100) if total > 0 else 0

    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'score_percentage': score_percentage,
        'results': results
    }


class QuizPDF(FPDF):
    """Custom PDF class for quiz results"""

    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Quiz Results Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_title_section(self, title: str, date_time: str, score: float, correct: int, total: int):
        """Add title section with score summary"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1)

        self.set_font('Arial', '', 11)
        self.cell(0, 8, f'Date: {date_time}', 0, 1)
        self.ln(2)

        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, f'Score: {score:.1f}% ({correct}/{total} correct)', 0, 1, 'C', True)
        self.ln(5)

    def add_question_result(self, num: int, result: Dict[str, Any]):
        """Add individual question result"""
        # Question number and correctness indicator
        self.set_font('Arial', 'B', 11)
        status = '[CORRECT]' if result['is_correct'] else '[INCORRECT]'
        color = (0, 150, 0) if result['is_correct'] else (200, 0, 0)

        self.set_text_color(*color)
        self.cell(0, 8, f"Question {num}: {status}", 0, 1)
        self.set_text_color(0, 0, 0)

        # Question text
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, f"Q: {result['question']}")

        # User answer
        self.set_font('Arial', 'I', 10)
        self.multi_cell(0, 6, f"Your answer: {result['user_answer']}")

        # Correct answer
        self.set_font('Arial', 'B', 10)
        self.multi_cell(0, 6, f"Correct answer: {result['correct_answer']}")

        # Explanation
        self.set_font('Arial', '', 9)
        self.set_text_color(50, 50, 150)
        self.multi_cell(0, 5, f"Explanation: {result['explanation']}")
        self.set_text_color(0, 0, 0)

        # Chapter reference
        if result['chapter'] != 'N/A':
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, f"Chapter: {result['chapter']}", 0, 1)

        self.ln(4)


def generate_pdf(grading_results: Dict[str, Any], quiz_title: str = "Quiz") -> bytes:
    """
    Generate PDF report of quiz results

    Args:
        grading_results: Results from grade_quiz function
        quiz_title: Title of the quiz

    Returns:
        PDF content as bytes
    """
    pdf = QuizPDF()
    pdf.add_page()

    # Add title section
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.add_title_section(
        quiz_title,
        date_time,
        grading_results['score_percentage'],
        grading_results['correct'],
        grading_results['total']
    )

    # Add each question result
    for i, result in enumerate(grading_results['results'], 1):
        # Add new page if needed
        if pdf.get_y() > 250:
            pdf.add_page()

        pdf.add_question_result(i, result)

    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin1')


def calculate_quiz_length(total_questions: int, percentage: int = None, custom_num: int = None) -> int:
    """
    Calculate number of questions based on percentage or custom number

    Args:
        total_questions: Total number of available questions
        percentage: Percentage of questions to use (10-100)
        custom_num: Custom number of questions

    Returns:
        Number of questions for the quiz
    """
    if custom_num is not None and custom_num > 0:
        return min(custom_num, total_questions)

    if percentage is not None:
        num = int(total_questions * percentage / 100)
        return max(1, min(num, total_questions))

    return total_questions
