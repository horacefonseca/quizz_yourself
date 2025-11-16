# Interactive Quiz Application

A flexible, interactive quiz application built with Streamlit that works with any YAML or Markdown formatted question bank.

## Features

- **Universal Format Support**: Works with any properly formatted YAML or Markdown question bank
- **Two Question Types**:
  - Multiple Choice (MC)
  - Open-ended questions
- **Flexible Quiz Length**: Choose percentage or custom number of questions
- **Random Sampling**: Questions randomly selected without repetition
- **Instant Grading**: Automatic grading with detailed feedback
- **PDF Export**: Download quiz results as a formatted PDF
- **Bundled Quizzes**:
  - **Complete Bank**: 250 comprehensive ML classification questions
  - **Quick Sample**: 40 essential ML questions for practice

## Project Structure

```
quiz_app/
├── app.py                                    # Main Streamlit application
├── utils.py                                  # Helper functions (YAML/MD loader, grading, PDF)
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
└── sample_quizzes/
    ├── ML2_Professor_Rico.yaml              # Quick sample YAML (40 questions)
    ├── ML_Classification_Complete_250.yaml  # Complete YAML bank (250 questions)
    ├── ML_Classification_Complete_250.md    # Complete MD bank (250 questions)
    └── Sample_Quiz.md                       # Sample markdown format (15 questions)
```

## Bundled Question Banks

The app includes two preset ML Classification quiz banks:

### 1. ML Classification Complete (250 questions)
**Filenames:** `ML_Classification_Complete_250.yaml` / `ML_Classification_Complete_250.md`

The most comprehensive option for thorough assessment (available in both YAML and Markdown formats):
- **Part A**: 125 Concept Questions covering:
  - Probability & Statistical Foundations (15)
  - Binary Classification Fundamentals (20)
  - Advanced Evaluation Metrics (20)
  - Class Imbalance & Advanced Techniques (15)
  - Multiclass Classification (15)
  - Decision Trees (20)
  - Ensemble Methods (20)

- **Part B**: 125 Coding Questions covering:
  - NumPy Basics (15)
  - Pandas Fundamentals (20)
  - Matplotlib Basics (15)
  - Scikit-Learn: Model Selection (20)
  - Scikit-Learn: Metrics (20)
  - Scikit-Learn: Preprocessing (15)
  - Scikit-Learn: Models (20)

**Best for:** Comprehensive exams, semester finals, certification prep

### 2. ML2 Professor Rico Sample (40 questions)
**Filename:** `ML2_Professor_Rico.yaml`

A curated selection of essential questions:
- Core ML concepts
- Key evaluation metrics
- Essential coding syntax
- Quick assessment format

**Best for:** Practice quizzes, quick review, class exercises

## Question Bank Formats

The app supports **two file formats**: YAML and Markdown.

### Format 1: YAML (.yaml, .yml)

Structured format ideal for programmatic generation.

#### Multiple Choice Question Example:
```yaml
- id: 1
  question: "What is the sigmoid output range?"
  type: "mc"
  options:
    - "A) 0–1"
    - "B) -1–1"
    - "C) 0–10"
  correct: "A"
  explanation: "Sigmoid always outputs 0–1."
  chapter: "1.4"
```

#### Open Question Example:
```yaml
- id: 2
  question: "Define Logistic Regression"
  type: "open"
  answer: "A model for binary classification mapping linear inputs to probabilities using sigmoid."
  explanation: "It models log-odds as a linear function."
  chapter: "4"
```

---

### Format 2: Markdown (.md)

Human-readable format ideal for manual creation and readability.

#### Multiple Choice Question Example:
```markdown
# Question 1

**Type:** mc

What is the sigmoid output range?

**Options:**
- A) 0–1
- B) -1–1
- C) 0–10

**Correct:** A

**Explanation:** Sigmoid always outputs 0–1.

**Chapter:** 1.4

---
```

#### Open Question Example:
```markdown
# Question 2

**Type:** open

Define Logistic Regression

**Answer:** A model for binary classification mapping linear inputs to probabilities using sigmoid.

**Explanation:** It models log-odds as a linear function.

**Chapter:** 4

---
```

#### Markdown Format Notes:
- Each question starts with `# Question {number}` or just `# Question`
- Separate questions with `---` (three dashes)
- Field names use bold markdown: `**Field Name:**`
- Options use markdown list format: `- A) option text`
- IDs are auto-assigned if not specified in header

---

### Required Fields (Both Formats):
- **id**: Unique question identifier (integer)
- **question**: Question text (string)
- **type**: Question type - either "mc" or "open"

### Type-Specific Required Fields:

**For MC questions:**
- **options**: List of answer choices (strings, typically formatted as "A) text", "B) text", etc.)
- **correct**: Correct answer letter (string, e.g., "A", "B", "C")

**For Open questions:**
- **answer**: Correct answer text (string)

### Optional Fields:
- **explanation**: Explanation of the correct answer (string)
- **chapter**: Chapter or topic reference (string)

## Installation & Local Setup

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

1. **Clone or download this repository**
   ```bash
   git clone <your-repo-url>
   cd quiz_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## Deployment to Streamlit Cloud

### Step 1: Prepare GitHub Repository

1. **Create a new GitHub repository**
   - Go to https://github.com/new
   - Name your repository (e.g., `quiz-app`)
   - Make it public
   - Don't initialize with README (we have our own)

2. **Upload your files**

   Option A: Using GitHub web interface
   - Click "uploading an existing file"
   - Drag and drop all files from your quiz_app folder
   - Maintain the folder structure (sample_quizzes subfolder)

   Option B: Using Git command line
   ```bash
   cd quiz_app
   git init
   git add .
   git commit -m "Initial commit - Quiz Application"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/quiz-app.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit https://streamlit.io/cloud
   - Click "Sign up" or "Sign in" with your GitHub account

2. **Create a new app**
   - Click "New app" button
   - Select your repository from the dropdown
   - Set the main file path: `app.py`
   - Choose a custom app URL (optional)

3. **Deploy**
   - Click "Deploy!"
   - Wait 2-5 minutes for deployment
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Step 3: Verify Deployment

1. Visit your app URL
2. Test with the bundled quiz
3. Try uploading a custom YAML file
4. Complete a quiz and download the PDF

## Usage Guide

### For Students/Users:

1. **Start the App**
   - Visit the app URL or run locally

2. **Select Question Bank**
   - Choose "Use bundled quiz" and select:
     - **ML Classification Complete (250 questions)** for comprehensive assessment
     - **ML2 Professor Rico Sample (40 questions)** for quick practice
   - OR Choose "Upload YAML file" to use your own questions

3. **Configure Quiz Length**
   - Use the percentage slider (10%-100%)
   - Or check "Use custom number" for exact count

4. **Take the Quiz**
   - Click "Start Quiz"
   - Answer all questions
   - Click "Submit Quiz"

5. **Review Results**
   - See your score and detailed feedback
   - Review explanations for all questions
   - Download PDF for your records

### For Instructors:

#### Adding New Quizzes

1. **Create your question bank file**
   - Choose YAML (.yaml) or Markdown (.md) format
   - Follow the format examples shown above
   - For YAML: Validate with a YAML validator
   - Save as `YourQuizName.yaml` or `YourQuizName.md`

2. **Add to repository**
   - Place in `sample_quizzes/` folder
   - Commit and push to GitHub
   - Streamlit Cloud will auto-redeploy

3. **Your quiz will appear**
   - In the "Select a quiz" dropdown
   - Users can select it from the sidebar

#### Creating Custom Question Banks

Tips for creating effective quizzes:

1. **Use consistent formatting**
   - For MC: Always use "A)", "B)", "C)", "D)" format
   - Keep IDs sequential
   - Include explanations for learning value

2. **Mix question types**
   - Use MC for factual recall
   - Use open questions for definitions/concepts

3. **Organize by topic**
   - Use chapter field for categorization
   - Helps students identify weak areas

4. **Test your question bank**
   - For YAML: Validate at https://www.yamllint.com
   - For Markdown: Check separator `---` and bold field syntax
   - Test upload in the app before sharing

## Troubleshooting

### Common Issues

**Question bank won't load**
- **For YAML**: Check for proper indentation (use spaces, not tabs), validate at yamllint.com
- **For Markdown**: Ensure questions are separated by `---`, field names use bold syntax `**Field:**`
- Ensure all required fields are present for both formats

**App crashes on deployment**
- Verify requirements.txt matches your local environment
- Check Streamlit Cloud logs for specific errors
- Ensure all files are committed to GitHub

**PDF download fails**
- This should work with fpdf library
- Check browser's download settings
- Try a different browser

**Questions not displaying**
- Verify YAML or Markdown structure matches examples in this README
- Check question type is either "mc" or "open"
- For MC questions: Ensure options list is properly formatted (4 options in YAML list or markdown bullets)

## Technical Details

### Dependencies

- **streamlit**: Web application framework
- **pyyaml**: YAML file parsing
- **fpdf**: PDF generation (no OS dependencies)

### Session State Variables

The app uses Streamlit's session state to maintain:
- `quiz_started`: Boolean flag
- `questions`: All loaded questions
- `quiz_questions`: Sampled questions for current quiz
- `user_answers`: Dictionary of user responses
- `quiz_submitted`: Submission status
- `grading_results`: Grading output

### Grading Logic

- **MC Questions**: Exact letter match (case-insensitive)
- **Open Questions**: Exact text match (case-insensitive)
- Score: (correct / total) × 100

## Customization

### Changing App Appearance

Edit `app.py`:
- Modify `st.set_page_config()` for title, icon, layout
- Adjust color schemes in metric displays
- Customize success/warning messages

### Modifying Grading

Edit `utils.py`:
- `grade_quiz()` function for scoring logic
- Implement fuzzy matching for open questions
- Add partial credit scoring

### PDF Formatting

Edit `utils.py`:
- `QuizPDF` class for layout changes
- Adjust fonts, colors, spacing
- Add logos or institutional branding

## License

This application is open-source and free to use for educational purposes.

## Support

For issues or questions:
1. Check this README
2. Review YAML format examples
3. Test with the bundled quiz first
4. Check Streamlit Cloud deployment logs

## Credits

Built with Streamlit for educational assessment and self-study.

Sample quiz covers Machine Learning II topics for MDC students.
