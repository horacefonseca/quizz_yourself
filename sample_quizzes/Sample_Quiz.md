# Machine Learning Sample Quiz - Markdown Format

This is a sample quiz in Markdown format demonstrating the supported syntax.

---

# Question 1

**Type:** mc

What is the range of valid probability values?

**Options:**
- A) Between -1 and 1
- B) Between 0 and 1
- C) Between 0 and 100
- D) Any positive number

**Correct:** B

**Explanation:** Probabilities must be between 0 (impossible) and 1 (certain).

**Chapter:** 1.1

---

# Question 2

**Type:** mc

What is the output range of the sigmoid function?

**Options:**
- A) (-∞, +∞)
- B) (0, 1)
- C) [0, 1]
- D) (-1, 1)

**Correct:** B

**Explanation:** Sigmoid squashes any input into the probability range between 0 and 1.

**Chapter:** 1.4

---

# Question 3

**Type:** open

Define binary classification.

**Answer:** Classification with exactly two possible class labels

**Explanation:** Binary classification has exactly two possible outcomes, like yes/no or positive/negative.

**Chapter:** 3

---

# Question 4

**Type:** mc

What distinguishes classification from regression?

**Options:**
- A) Number of features used
- B) Classification predicts categorical outcomes, regression predicts continuous
- C) Classification is always more accurate
- D) Regression cannot use multiple features

**Correct:** B

**Explanation:** Classification predicts discrete categories; regression predicts continuous numerical values.

**Chapter:** 3

---

# Question 5

**Type:** mc

Define precision (positive predictive value).

**Options:**
- A) TP / (TP + FN)
- B) TP / (TP + FP)
- C) TN / (TN + FP)
- D) (TP + TN) / Total

**Correct:** B

**Explanation:** Precision measures how many positive predictions were actually correct.

**Chapter:** 6

---

# Question 6

**Type:** mc

Define recall (sensitivity, true positive rate).

**Options:**
- A) TP / (TP + FP)
- B) TP / (TP + FN)
- C) TN / (TN + FN)
- D) FP / (FP + TN)

**Correct:** B

**Explanation:** Recall measures what fraction of actual positives were correctly identified.

**Chapter:** 6

---

# Question 7

**Type:** open

What is cross-validation?

**Answer:** Splitting data into folds for robust evaluation

**Explanation:** Cross-validation splits data into multiple folds for more robust performance estimation.

**Chapter:** 7

---

# Question 8

**Type:** mc

What is the regularization parameter C in logistic regression?

**Options:**
- A) Number of classes
- B) Inverse of regularization strength
- C) Learning rate
- D) Number of iterations

**Correct:** B

**Explanation:** C controls regularization strength; smaller C means stronger regularization.

**Chapter:** 7

---

# Question 9

**Type:** mc

What is multiclass classification?

**Options:**
- A) Classification with multiple features
- B) Classification with more than two classes
- C) Classification using multiple models
- D) Binary classification repeated

**Correct:** B

**Explanation:** Classification problem with three or more mutually exclusive class categories.

**Chapter:** 13

---

# Question 10

**Type:** mc

How does a decision tree make predictions?

**Options:**
- A) Using linear equations
- B) Following if-then rules from root to leaf
- C) Calculating probabilities only
- D) Random selection

**Correct:** B

**Explanation:** Follows if-then decision rules from root to leaf node containing prediction.

**Chapter:** 16

---

# Question 11

**Type:** open

Define pruning in decision trees.

**Answer:** Removing branches to reduce overfitting

**Explanation:** Removing tree branches to reduce complexity and prevent overfitting.

**Chapter:** 17

---

# Question 12

**Type:** mc

What is ensemble learning?

**Options:**
- A) Training one large model
- B) Combining predictions from multiple models for better performance
- C) Using multiple features
- D) Training on multiple datasets separately

**Correct:** B

**Explanation:** Combining predictions from multiple models to improve overall performance.

**Chapter:** 18

---

# Question 13

**Type:** mc

How does Random Forest work?

**Options:**
- A) One deep decision tree
- B) Ensemble of decision trees with random feature subsets
- C) Linear combination of features
- D) Sequential tree building

**Correct:** B

**Explanation:** Ensemble of decision trees using bootstrap samples and random feature subsets.

**Chapter:** 18

---

# Question 14

**Type:** mc

How do you import NumPy with standard alias?

**Options:**
- A) import numpy
- B) import numpy as np
- C) from numpy import *
- D) import np

**Correct:** B

**Explanation:** Standard convention: import numpy as np for brevity.

**Chapter:** NumPy

---

# Question 15

**Type:** mc

How do you import pandas with standard alias?

**Options:**
- A) import pandas
- B) import pandas as pd
- C) from pandas import *
- D) import pd

**Correct:** B

**Explanation:** Standard convention: import pandas as pd for brevity.

**Chapter:** Pandas

---

End of Quiz
