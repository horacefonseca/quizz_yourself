# Machine Learning Classification - Complete 250 Question Bank

Comprehensive assessment covering all ML classification topics.
Part A: 125 Concept Questions | Part B: 125 Coding Questions

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

**Chapter:** Chapter 1.1

---

# Question 2

**Type:** mc

If P(event A) = 0.30, what is P(not A)?

**Options:**
- A) 0.30
- B) 0.50
- C) 0.70
- D) 0.90

**Correct:** C

**Explanation:** Complement rule: P(not A) = 1 - P(A) = 0.70.

**Chapter:** Chapter 1.1

---

# Question 3

**Type:** mc

What does it mean if two events are mutually exclusive?

**Options:**
- A) They always occur together
- B) They cannot occur at the same time
- C) They are independent events
- D) They have equal probability

**Correct:** B

**Explanation:** Mutually exclusive events cannot happen simultaneously, like heads and tails.

**Chapter:** Chapter 1.1

---

# Question 4

**Type:** mc

What is conditional probability P(A|B)?

**Options:**
- A) Probability of A multiplied by B
- B) Probability of A given that B has occurred
- C) Probability of A plus B
- D) Probability of A divided by 2

**Correct:** B

**Explanation:** Conditional probability measures the probability of A occurring given B occurred.

**Chapter:** Chapter 1.2

---

# Question 5

**Type:** mc

In classification, what does P(Y=1 | X=x) represent?

**Options:**
- A) The feature value x
- B) The probability of class 1 given features x
- C) The total number of samples
- D) The classification error

**Correct:** B

**Explanation:** It represents the probability of positive class given specific feature values.

**Chapter:** Chapter 1.2

---

# Question 6

**Type:** mc

Define "odds" in terms of probability.

**Options:**
- A) 1 - P
- B) P / (1 - P)
- C) P * (1 - P)
- D) log(P)

**Correct:** B

**Explanation:** Odds are calculated as probability of success divided by probability of failure.

**Chapter:** Chapter 1.3

---

# Question 7

**Type:** mc

If P = 0.80, what are the odds?

**Options:**
- A) 0.20
- B) 0.80
- C) 4.0
- D) 1.25

**Correct:** C

**Explanation:** Odds = 0.80 / (1 - 0.80) = 0.80 / 0.20 = 4.0.

**Chapter:** Chapter 1.3

---

# Question 8

**Type:** mc

What is the range of possible odds values?

**Options:**
- A) Between 0 and 1
- B) Between -∞ and +∞
- C) Between 0 and +∞
- D) Between 0 and 100

**Correct:** C

**Explanation:** Odds range from 0 to infinity; zero means impossible, infinity means certain.

**Chapter:** Chapter 1.3

---

# Question 9

**Type:** mc

What is the logit (log-odds) when P = 0.5?

**Options:**
- A) -1
- B) 0
- C) 0.5
- D) 1

**Correct:** B

**Explanation:** When P = 0.5, odds = 1, and log(1) = 0.

**Chapter:** Chapter 1.3

---

# Question 10

**Type:** mc

Why do we use log-odds instead of probability in logistic regression?

**Options:**
- A) Log-odds are easier to calculate
- B) Log-odds allow for a linear relationship with predictors
- C) Log-odds are always positive
- D) Probabilities cannot be modeled

**Correct:** B

**Explanation:** Log-odds can be any real number, allowing linear modeling with predictors.

**Chapter:** Chapter 1.3

---

# Question 11

**Type:** mc

What is the output range of the sigmoid function?

**Options:**
- A) (-∞, +∞)
- B) (0, 1)
- C) [0, 1]
- D) (-1, 1)

**Correct:** B

**Explanation:** Sigmoid squashes any input into the probability range between 0 and 1.

**Chapter:** Chapter 1.4

---

# Question 12

**Type:** mc

What is σ(0) where σ is the sigmoid function?

**Options:**
- A) 0
- B) 0.5
- C) 1
- D) undefined

**Correct:** B

**Explanation:** Sigmoid of zero equals 0.5, representing equal probability for both classes.

**Chapter:** Chapter 1.4

---

# Question 13

**Type:** mc

Describe the shape of the sigmoid curve.

**Options:**
- A) Linear increasing
- B) U-shaped
- C) S-shaped
- D) Bell-shaped

**Correct:** C

**Explanation:** Sigmoid has S-shape: flat at extremes, steep in middle around zero.

**Chapter:** Chapter 1.4

---

# Question 14

**Type:** mc

What is the relationship between sigmoid and logit?

**Options:**
- A) They are the same function
- B) Sigmoid is the inverse of logit
- C) Logit is the derivative of sigmoid
- D) They are unrelated

**Correct:** B

**Explanation:** Sigmoid converts log-odds to probability; logit does the inverse transformation.

**Chapter:** Chapter 1.4

---

# Question 15

**Type:** mc

Why is the sigmoid function S-shaped rather than linear?

**Options:**
- A) To make calculations easier
- B) To constrain output to probability range (0,1)
- C) To increase model complexity
- D) It's purely aesthetic

**Correct:** B

**Explanation:** S-shape ensures output stays within valid probability range of 0 to 1.

**Chapter:** Chapter 1.4

---

# Question 16

**Type:** mc

What distinguishes classification from regression?

**Options:**
- A) Number of features used
- B) Classification predicts categorical outcomes, regression predicts continuous
- C) Classification is always more accurate
- D) Regression cannot use multiple features

**Correct:** B

**Explanation:** Classification predicts discrete categories; regression predicts continuous numerical values.

**Chapter:** Chapter 3

---

# Question 17

**Type:** mc

Define binary classification.

**Options:**
- A) Classification with two features
- B) Classification with exactly two possible class labels
- C) Classification using binary code
- D) Classification of 0s and 1s only

**Correct:** B

**Explanation:** Binary classification has exactly two possible outcomes, like yes/no or positive/negative.

**Chapter:** Chapter 3

---

# Question 18

**Type:** mc

What is a class label?

**Options:**
- A) The feature importance score
- B) The categorical outcome we want to predict
- C) The model accuracy
- D) The number of classes

**Correct:** B

**Explanation:** Class label is the categorical outcome variable we're trying to predict.

**Chapter:** Chapter 3

---

# Question 19

**Type:** mc

What is class imbalance?

**Options:**
- A) When features have different scales
- B) When one class has significantly more samples than another
- C) When the model is biased
- D) When classes overlap

**Correct:** B

**Explanation:** Class imbalance means one class has significantly more samples than the other.

**Chapter:** Chapter 3

---

# Question 20

**Type:** mc

Why is the naive classifier (always predicting majority class) problematic?

**Options:**
- A) It's computationally expensive
- B) It fails to identify minority class instances
- C) It requires too much data
- D) It cannot be implemented

**Correct:** B

**Explanation:** It ignores minority class completely, achieving high accuracy but poor real performance.

**Chapter:** Chapter 3

---

# Question 21

**Type:** mc

Why does linear regression fail for binary outcomes?

**Options:**
- A) It's too slow
- B) It can predict values outside [0,1] range
- C) It requires categorical inputs
- D) It cannot handle two classes

**Correct:** B

**Explanation:** Linear regression can output values outside the valid probability range of 0-1.

**Chapter:** Chapter 4

---

# Question 22

**Type:** mc

What function does logistic regression model as linear?

**Options:**
- A) The probability directly
- B) The log-odds (logit)
- C) The error term
- D) The class labels

**Correct:** B

**Explanation:** Logistic regression models log-odds as a linear combination of predictors.

**Chapter:** Chapter 4

---

# Question 23

**Type:** mc

Write the logistic regression equation in terms of log-odds.

**Options:**
- A) P(Y=1) = β₀ + β₁X
- B) log(P) = β₀ + β₁X
- C) log(P/(1-P)) = β₀ + β₁X
- D) P/(1-P) = β₀ + β₁X

**Correct:** C

**Explanation:** This is the logit form: log(odds) = log(P/(1-P)) = β₀ + β₁X.

**Chapter:** Chapter 4

---

# Question 24

**Type:** mc

What does the coefficient β₁ represent in logistic regression?

**Options:**
- A) The probability of class 1
- B) The change in log-odds per unit change in X
- C) The model accuracy
- D) The intercept value

**Correct:** B

**Explanation:** β₁ represents change in log-odds for each one-unit increase in X.

**Chapter:** Chapter 4

---

# Question 25

**Type:** mc

How is maximum likelihood estimation used in logistic regression?

**Options:**
- A) To split the data
- B) To find coefficients that maximize probability of observed data
- C) To calculate accuracy
- D) To normalize features

**Correct:** B

**Explanation:** MLE finds coefficients that maximize the likelihood of observing the training data.

**Chapter:** Chapter 4

---

# Question 26

**Type:** mc

What does a positive coefficient mean in logistic regression?

**Options:**
- A) The feature is irrelevant
- B) Increasing the feature increases log-odds of class 1
- C) The model is overfitting
- D) The feature should be removed

**Correct:** B

**Explanation:** Positive coefficient means higher feature values increase probability of positive class.

**Chapter:** Chapter 5

---

# Question 27

**Type:** mc

Define odds ratio.

**Options:**
- A) The ratio of two probabilities
- B) The ratio of odds for different feature values
- C) The ratio of true to false predictions
- D) The ratio of training to test accuracy

**Correct:** B

**Explanation:** Odds ratio compares odds at two different feature values.

**Chapter:** Chapter 5

---

# Question 28

**Type:** mc

How do you calculate odds ratio from a coefficient?

**Options:**
- A) β₁
- B) log(β₁)
- C) exp(β₁)
- D) β₁²

**Correct:** C

**Explanation:** Exponentiating the coefficient gives the multiplicative change in odds.

**Chapter:** Chapter 5

---

# Question 29

**Type:** mc

What does an odds ratio of 2.5 mean?

**Options:**
- A) 25% probability increase
- B) The odds are 2.5 times higher for 1 unit increase in feature
- C) The accuracy is 2.5 times better
- D) There are 2.5 classes

**Correct:** B

**Explanation:** One unit increase in feature multiplies the odds by 2.5.

**Chapter:** Chapter 5

---

# Question 30

**Type:** mc

Why are standardized coefficients useful for interpretation?

**Options:**
- A) They make the model faster
- B) They allow comparison of relative feature importance
- C) They improve accuracy
- D) They are required for logistic regression

**Correct:** B

**Explanation:** Standardization allows fair comparison of effect sizes across different feature scales.

**Chapter:** Chapter 5

---

# Question 31

**Type:** mc

What is a confusion matrix?

**Options:**
- A) A matrix of model parameters
- B) A table showing actual vs predicted classifications
- C) A correlation matrix
- D) A matrix of feature importances

**Correct:** B

**Explanation:** Confusion matrix shows counts of actual vs predicted classes in a table.

**Chapter:** Chapter 6

---

# Question 32

**Type:** mc

Define True Positive (TP).

**Options:**
- A) Correctly predicted negative cases
- B) Correctly predicted positive cases
- C) Incorrectly predicted positive cases
- D) Incorrectly predicted negative cases

**Correct:** B

**Explanation:** True Positive: model correctly predicted the positive class.

**Chapter:** Chapter 6

---

# Question 33

**Type:** mc

Define False Positive (FP).

**Options:**
- A) Actual positive predicted as negative
- B) Actual negative predicted as positive
- C) Actual positive predicted as positive
- D) Actual negative predicted as negative

**Correct:** B

**Explanation:** False Positive: model incorrectly predicted positive when actually negative (Type I error).

**Chapter:** Chapter 6

---

# Question 34

**Type:** mc

What is accuracy and why can it be misleading?

**Options:**
- A) Accuracy is always the best metric
- B) Accuracy can be high even with poor minority class performance
- C) Accuracy measures only false positives
- D) Accuracy cannot be calculated for binary classification

**Correct:** B

**Explanation:** Accuracy can be misleading with imbalanced data; high from majority class only.

**Chapter:** Chapter 6

---

# Question 35

**Type:** mc

Define precision (positive predictive value).

**Options:**
- A) TP / (TP + FN)
- B) TP / (TP + FP)
- C) TN / (TN + FP)
- D) (TP + TN) / Total

**Correct:** B

**Explanation:** Precision measures how many positive predictions were actually correct.

**Chapter:** Chapter 6

---

# Question 36

**Type:** mc

Define recall (sensitivity, true positive rate).

**Options:**
- A) TP / (TP + FP)
- B) TP / (TP + FN)
- C) TN / (TN + FN)
- D) FP / (FP + TN)

**Correct:** B

**Explanation:** Recall measures what fraction of actual positives were correctly identified.

**Chapter:** Chapter 6

---

# Question 37

**Type:** mc

Define specificity (true negative rate).

**Options:**
- A) TP / (TP + FN)
- B) TN / (TN + FP)
- C) TN / (TN + FN)
- D) FN / (FN + TP)

**Correct:** B

**Explanation:** Specificity measures what fraction of actual negatives were correctly identified.

**Chapter:** Chapter 6

---

# Question 38

**Type:** mc

What is the F1-score and when is it useful?

**Options:**
- A) Average of precision and recall
- B) Harmonic mean of precision and recall, useful for imbalanced data
- C) Product of precision and recall
- D) Maximum of precision and recall

**Correct:** B

**Explanation:** F1-score is harmonic mean balancing precision and recall for imbalanced datasets.

**Chapter:** Chapter 6

---

# Question 39

**Type:** mc

What does the F1-score balance?

**Options:**
- A) Accuracy and speed
- B) Precision and recall
- C) Training and test performance
- D) True positives and true negatives

**Correct:** B

**Explanation:** F1-score provides single metric combining both precision and recall performance.

**Chapter:** Chapter 6

---

# Question 40

**Type:** mc

What is the ROC curve?

**Options:**
- A) Plot of accuracy vs threshold
- B) Plot of TPR vs FPR at different thresholds
- C) Plot of precision vs recall
- D) Plot of training vs test error

**Correct:** B

**Explanation:** ROC plots true positive rate vs false positive rate across all thresholds.

**Chapter:** Chapter 6

---

# Question 41

**Type:** mc

What is AUC (Area Under Curve)?

**Options:**
- A) Total model accuracy
- B) Area under ROC curve, measure of classification ability
- C) Average of all metrics
- D) Number of correct predictions

**Correct:** B

**Explanation:** AUC measures overall classification ability; higher is better, ranges 0 to 1.

**Chapter:** Chapter 6

---

# Question 42

**Type:** mc

What does AUC = 0.5 indicate?

**Options:**
- A) Perfect classification
- B) Random guessing / no discrimination
- C) 50% accuracy
- D) Excellent performance

**Correct:** B

**Explanation:** AUC of 0.5 means model performs no better than random chance.

**Chapter:** Chapter 6

---

# Question 43

**Type:** mc

What does AUC = 1.0 indicate?

**Options:**
- A) Random performance
- B) Perfect classification
- C) 50% accuracy
- D) Model overfitting

**Correct:** B

**Explanation:** Perfect classifier with AUC of 1.0 separates all classes correctly.

**Chapter:** Chapter 6

---

# Question 44

**Type:** mc

When should you prioritize precision over recall?

**Options:**
- A) When false positives are more costly than false negatives
- B) When false negatives are more costly
- C) When classes are balanced
- D) Always prioritize precision

**Correct:** A

**Explanation:** When false positives are costly, like spam detection or fraud alerts.

**Chapter:** Chapter 6

---

# Question 45

**Type:** mc

When should you prioritize recall over precision?

**Options:**
- A) When false positives are more costly
- B) When false negatives are more costly (e.g., disease detection)
- C) When accuracy is high
- D) Never prioritize recall

**Correct:** B

**Explanation:** When missing positives is costly, like disease detection or safety monitoring.

**Chapter:** Chapter 6

---

# Question 46

**Type:** mc

What is a hyperparameter?

**Options:**
- A) A parameter learned during training
- B) A parameter set before training that controls learning
- C) The model's prediction
- D) The training data size

**Correct:** B

**Explanation:** Hyperparameter is set before training and controls the learning algorithm behavior.

**Chapter:** Chapter 7

---

# Question 47

**Type:** mc

What is cross-validation?

**Options:**
- A) Training on entire dataset
- B) Splitting data into folds for robust evaluation
- C) Testing on training data
- D) Validating feature names

**Correct:** B

**Explanation:** Cross-validation splits data into multiple folds for more robust performance estimation.

**Chapter:** Chapter 7

---

# Question 48

**Type:** mc

What is K-fold cross-validation?

**Options:**
- A) Using K features
- B) Splitting data into K folds, training on K-1, testing on 1
- C) Running K models
- D) Training for K epochs

**Correct:** B

**Explanation:** Data divided into K parts; train on K-1, test on 1, repeat K times.

**Chapter:** Chapter 7

---

# Question 49

**Type:** mc

Why use cross-validation instead of a single train/test split?

**Options:**
- A) It's faster
- B) It provides more robust performance estimates
- C) It requires less data
- D) It improves accuracy

**Correct:** B

**Explanation:** Reduces variance in performance estimates and uses all data for testing.

**Chapter:** Chapter 7

---

# Question 50

**Type:** mc

What is GridSearchCV?

**Options:**
- A) A plotting function
- B) Exhaustive search over hyperparameter grid with CV
- C) A data preprocessing tool
- D) A feature selection method

**Correct:** B

**Explanation:** Exhaustively searches hyperparameter combinations using cross-validation to find best settings.

**Chapter:** Chapter 7

---

# Question 51

**Type:** mc

What is the regularization parameter C in logistic regression?

**Options:**
- A) Number of classes
- B) Inverse of regularization strength
- C) Learning rate
- D) Number of iterations

**Correct:** B

**Explanation:** C controls regularization strength; smaller C means stronger regularization.

**Chapter:** Chapter 7

---

# Question 52

**Type:** mc

What happens when C is very small?

**Options:**
- A) No regularization
- B) Strong regularization, simpler model
- C) Model becomes more complex
- D) Training fails

**Correct:** B

**Explanation:** Small C applies strong regularization, shrinking coefficients toward zero for simplicity.

**Chapter:** Chapter 7

---

# Question 53

**Type:** mc

What happens when C is very large?

**Options:**
- A) Strong regularization
- B) Weak regularization, more complex model
- C) Model cannot fit
- D) All coefficients become zero

**Correct:** B

**Explanation:** Large C applies weak regularization, allowing larger coefficients and more complexity.

**Chapter:** Chapter 7

---

# Question 54

**Type:** mc

What is the purpose of stratification in train_test_split?

**Options:**
- A) To randomize data
- B) To preserve class proportions in splits
- C) To increase dataset size
- D) To normalize features

**Correct:** B

**Explanation:** Stratification maintains the same class proportions in both train and test sets.

**Chapter:** Chapter 7

---

# Question 55

**Type:** mc

Why set random_state in train_test_split?

**Options:**
- A) To improve accuracy
- B) To ensure reproducibility of splits
- C) To increase randomness
- D) It's required for the function to work

**Correct:** B

**Explanation:** Random state ensures same split each run for reproducible and debuggable results.

**Chapter:** Chapter 7

---

# Question 56

**Type:** mc

What is class imbalance?

**Options:**
- A) Features with different scales
- B) Unequal number of samples across classes
- C) Model bias
- D) Overlapping classes

**Correct:** B

**Explanation:** One class has significantly fewer samples than others in the dataset.

**Chapter:** Chapter 8

---

# Question 57

**Type:** mc

What does class_weight='balanced' do?

**Options:**
- A) Removes minority class
- B) Automatically adjusts weights inversely proportional to class frequencies
- C) Balances feature scales
- D) Creates equal-sized classes

**Correct:** B

**Explanation:** Automatically assigns higher weights to minority class to balance importance.

**Chapter:** Chapter 8

---

# Question 58

**Type:** mc

How does class weighting affect the model?

**Options:**
- A) Increases accuracy only
- B) Penalizes misclassification of minority class more
- C) Removes all majority class samples
- D) Has no effect on predictions

**Correct:** B

**Explanation:** Increases penalty for minority class misclassification, improving minority class recall.

**Chapter:** Chapter 8

---

# Question 59

**Type:** mc

What is the probability threshold in classification?

**Options:**
- A) Minimum training accuracy
- B) Cutoff for converting probabilities to class labels (default 0.5)
- C) Maximum number of iterations
- D) The regularization strength

**Correct:** B

**Explanation:** Probability threshold determines cutoff for converting probabilities to class predictions.

**Chapter:** Chapter 8

---

# Question 60

**Type:** mc

Why might you change the default 0.5 threshold?

**Options:**
- A) To improve speed
- B) To adjust precision-recall trade-off for specific needs
- C) It must always be 0.5
- D) To reduce model complexity

**Correct:** B

**Explanation:** Adjust threshold to favor precision or recall based on specific problem needs.

**Chapter:** Chapter 8

---

# Question 61

**Type:** mc

What is SMOTE?

**Options:**
- A) A regularization technique
- B) Synthetic Minority Over-sampling Technique
- C) A feature selection method
- D) A type of neural network

**Correct:** B

**Explanation:** SMOTE generates synthetic minority class samples using nearest neighbor interpolation.

**Chapter:** Chapter 8

---

# Question 62

**Type:** mc

What are the trade-offs of oversampling the minority class?

**Options:**
- A) No trade-offs, always beneficial
- B) Risk of overfitting to minority class
- C) Decreases training time
- D) Removes all bias

**Correct:** B

**Explanation:** Creating synthetic samples may lead to overfitting on minority class patterns.

**Chapter:** Chapter 8

---

# Question 63

**Type:** mc

What are the trade-offs of undersampling the majority class?

**Options:**
- A) Always improves performance
- B) Loss of potentially useful information
- C) Increases dataset size
- D) No trade-offs

**Correct:** B

**Explanation:** Removing majority samples discards potentially useful information from the dataset.

**Chapter:** Chapter 8

---

# Question 64

**Type:** mc

How does changing threshold affect precision and recall?

**Options:**
- A) They always move together
- B) Increasing threshold typically increases precision, decreases recall
- C) Threshold doesn't affect metrics
- D) Only affects accuracy

**Correct:** B

**Explanation:** Higher threshold increases precision but decreases recall; opposite for lower threshold.

**Chapter:** Chapter 8

---

# Question 65

**Type:** mc

What is the precision-recall trade-off?

**Options:**
- A) Precision and recall are always equal
- B) Improving one often decreases the other
- C) They are independent metrics
- D) Trade-off only exists in multiclass

**Correct:** B

**Explanation:** Improving precision often reduces recall and vice versa; must find balance.

**Chapter:** Chapter 8

---

# Question 66

**Type:** mc

Why standardize features for logistic regression?

**Options:**
- A) It's required for the algorithm to work
- B) For comparable coefficient magnitudes and better convergence
- C) To increase accuracy
- D) To reduce the number of features

**Correct:** B

**Explanation:** Makes coefficients comparable and improves optimization convergence in gradient descent.

**Chapter:** Chapter 9

---

# Question 67

**Type:** mc

What does StandardScaler do?

**Options:**
- A) Scales features to [0,1]
- B) Transforms features to mean=0, std=1
- C) Removes outliers
- D) Selects important features

**Correct:** B

**Explanation:** Transforms each feature to have mean of zero and standard deviation of one.

**Chapter:** Chapter 9

---

# Question 68

**Type:** mc

What is the difference between normalization and standardization?

**Options:**
- A) They are the same
- B) Normalization scales to [0,1], standardization to mean=0 std=1
- C) Normalization is better
- D) Standardization is deprecated

**Correct:** B

**Explanation:** Normalization scales to 0-1 range; standardization centers at mean with unit variance.

**Chapter:** Chapter 9

---

# Question 69

**Type:** mc

Should you fit the scaler on test data?

**Options:**
- A) Yes, always fit on test data
- B) No, only fit on training data to avoid data leakage
- C) Fit on both training and test
- D) It doesn't matter

**Correct:** B

**Explanation:** Prevents data leakage; only training data should inform preprocessing parameters.

**Chapter:** Chapter 9

---

# Question 70

**Type:** mc

Why use pipelines for preprocessing?

**Options:**
- A) To make code slower
- B) To ensure consistent preprocessing and prevent data leakage
- C) Pipelines are required by scikit-learn
- D) To increase model complexity

**Correct:** B

**Explanation:** Ensures preprocessing steps applied consistently and prevents data leakage errors.

**Chapter:** Chapter 9

---

# Question 71

**Type:** mc

What is multiclass classification?

**Options:**
- A) Classification with multiple features
- B) Classification with more than two classes
- C) Classification using multiple models
- D) Binary classification repeated

**Correct:** B

**Explanation:** Classification problem with three or more mutually exclusive class categories.

**Chapter:** Chapter 13

---

# Question 72

**Type:** mc

What is the difference between binary and multiclass?

**Options:**
- A) Binary uses two features, multiclass uses more
- B) Binary has two classes, multiclass has three or more
- C) Binary is more accurate
- D) No difference

**Correct:** B

**Explanation:** Binary has two classes; multiclass has three or more possible categories.

**Chapter:** Chapter 13

---

# Question 73

**Type:** mc

What is one-vs-rest (OvR) strategy?

**Options:**
- A) Train one model for all classes
- B) Train one binary classifier per class vs all others
- C) Train models in sequence
- D) Use only one feature

**Correct:** B

**Explanation:** Trains one binary classifier per class versus all other classes combined.

**Chapter:** Chapter 13

---

# Question 74

**Type:** mc

What is one-vs-one (OvO) strategy?

**Options:**
- A) One model for all pairs of classes
- B) One model for each class
- C) One model total
- D) One feature per class

**Correct:** A

**Explanation:** Trains one binary classifier for each pair of classes.

**Chapter:** Chapter 13

---

# Question 75

**Type:** mc

What is multinomial logistic regression?

**Options:**
- A) Multiple binary logistic regressions
- B) Direct multiclass extension using softmax
- C) A type of neural network
- D) Cannot handle more than 2 classes

**Correct:** B

**Explanation:** Direct multiclass extension using softmax function for probability distribution.

**Chapter:** Chapter 13

---

# Question 76

**Type:** mc

How many sets of coefficients in multiclass LR with K classes?

**Options:**
- A) 1
- B) K-1
- C) K
- D) 2K

**Correct:** C

**Explanation:** K coefficient sets, one for each class in multiclass logistic regression.

**Chapter:** Chapter 14

---

# Question 77

**Type:** mc

How does multiclass LR make predictions?

**Options:**
- A) Randomly selects a class
- B) Calculates probabilities for all classes, selects highest
- C) Uses majority voting
- D) Predicts first class only

**Correct:** B

**Explanation:** Computes class probabilities using softmax, predicts class with highest probability.

**Chapter:** Chapter 14

---

# Question 78

**Type:** mc

What is the softmax function?

**Options:**
- A) A type of regularization
- B) Generalizes sigmoid to multiple classes, outputs probability distribution
- C) A loss function
- D) A feature scaling method

**Correct:** B

**Explanation:** Softmax converts raw scores into probability distribution summing to one.

**Chapter:** Chapter 14

---

# Question 79

**Type:** mc

How do you interpret multiclass confusion matrix?

**Options:**
- A) Same as binary, just 2x2
- B) K×K matrix where entry (i,j) shows actual class i predicted as j
- C) Cannot interpret multiclass confusion matrix
- D) Only diagonal matters

**Correct:** B

**Explanation:** Diagonal shows correct predictions; off-diagonal shows which classes are confused.

**Chapter:** Chapter 14

---

# Question 80

**Type:** mc

What is macro-average vs weighted-average in multiclass metrics?

**Options:**
- A) They are the same
- B) Macro: unweighted mean across classes; Weighted: by class support
- C) Macro is always better
- D) Weighted ignores minority classes

**Correct:** B

**Explanation:** Macro averages metrics equally; weighted averages by number of samples per class.

**Chapter:** Chapter 14

---

# Question 81

**Type:** mc

What is the Iris dataset used for?

**Options:**
- A) Binary classification
- B) 3-class flower species classification
- C) Regression
- D) Clustering only

**Correct:** B

**Explanation:** Classic dataset for 3-class flower species classification using petal measurements.

**Chapter:** Chapter 15

---

# Question 82

**Type:** mc

How many classes in the Wine dataset?

**Options:**
- A) 2
- B) 3
- C) 5
- D) 10

**Correct:** B

**Explanation:** Wine dataset contains three different wine cultivar classes.

**Chapter:** Chapter 15

---

# Question 83

**Type:** mc

What is the Digits dataset?

**Options:**
- A) Binary classification of digits
- B) 10-class handwritten digit recognition (0-9)
- C) Regression on numbers
- D) Only classifies 0 and 1

**Correct:** B

**Explanation:** Dataset of 8x8 pixel handwritten digits for 10-class classification.

**Chapter:** Chapter 15

---

# Question 84

**Type:** mc

Why is multiclass classification harder than binary?

**Options:**
- A) It's actually easier
- B) More classes means more complex decision boundaries
- C) Binary classification doesn't work
- D) No difference in difficulty

**Correct:** B

**Explanation:** More classes mean more complex decision boundaries and confusion between classes.

**Chapter:** Chapter 15

---

# Question 85

**Type:** mc

When would you use multiclass instead of multiple binary classifiers?

**Options:**
- A) Never use multiclass
- B) When classes are mutually exclusive and exhaustive
- C) Always use binary classifiers
- D) Only for 2 classes

**Correct:** B

**Explanation:** When outcomes are mutually exclusive categories, not independent binary decisions.

**Chapter:** Chapter 15

---

# Question 86

**Type:** mc

How does a decision tree make predictions?

**Options:**
- A) Using linear equations
- B) Following if-then rules from root to leaf
- C) Calculating probabilities only
- D) Random selection

**Correct:** B

**Explanation:** Follows if-then decision rules from root to leaf node containing prediction.

**Chapter:** Chapter 16

---

# Question 87

**Type:** mc

What is a split in a decision tree?

**Options:**
- A) Dividing training and test data
- B) Decision point that partitions data based on a feature
- C) Removing outliers
- D) Combining features

**Correct:** B

**Explanation:** Split divides data into subsets based on a feature threshold.

**Chapter:** Chapter 16

---

# Question 88

**Type:** mc

What is a leaf node?

**Options:**
- A) The root of the tree
- B) Terminal node containing class prediction
- C) A split point
- D) The entire tree

**Correct:** B

**Explanation:** Terminal node that contains the final class prediction or value.

**Chapter:** Chapter 16

---

# Question 89

**Type:** mc

What is tree depth?

**Options:**
- A) Number of features
- B) Longest path from root to leaf
- C) Number of samples
- D) Total number of nodes

**Correct:** B

**Explanation:** Maximum number of splits from root to any leaf node.

**Chapter:** Chapter 16

---

# Question 90

**Type:** mc

What is the Gini impurity?

**Options:**
- A) Measure of tree size
- B) Measure of class mixture/impurity in a node
- C) Number of mistakes
- D) Training accuracy

**Correct:** B

**Explanation:** Gini measures probability of incorrectly classifying a randomly chosen element.

**Chapter:** Chapter 16

---

# Question 91

**Type:** mc

What is entropy in decision trees?

**Options:**
- A) Tree depth
- B) Measure of disorder/uncertainty in class labels
- C) Number of splits
- D) Prediction accuracy

**Correct:** B

**Explanation:** Entropy measures uncertainty or disorder in the class distribution.

**Chapter:** Chapter 16

---

# Question 92

**Type:** mc

What is information gain?

**Options:**
- A) Number of new samples
- B) Reduction in entropy from a split
- C) Increase in accuracy
- D) Tree growth rate

**Correct:** B

**Explanation:** Reduction in entropy achieved by splitting on a particular feature.

**Chapter:** Chapter 16

---

# Question 93

**Type:** mc

How does a tree choose which feature to split on?

**Options:**
- A) Random selection
- B) Feature with highest information gain or lowest impurity
- C) First feature in dataset
- D) Most correlated feature

**Correct:** B

**Explanation:** Chooses feature that maximizes information gain or minimizes impurity after split.

**Chapter:** Chapter 16

---

# Question 94

**Type:** mc

What is overfitting in decision trees?

**Options:**
- A) Tree is too small
- B) Tree memorizes training data, poor generalization
- C) Perfect performance
- D) Tree cannot make predictions

**Correct:** B

**Explanation:** Tree learns training data too specifically, capturing noise instead of patterns.

**Chapter:** Chapter 16

---

# Question 95

**Type:** mc

Why do deep trees tend to overfit?

**Options:**
- A) They are too simple
- B) They create overly specific rules for training data
- C) They cannot learn patterns
- D) Deep trees are always better

**Correct:** B

**Explanation:** Deep trees create overly specific rules that don't generalize to new data.

**Chapter:** Chapter 16

---

# Question 96

**Type:** mc

What is pruning?

**Options:**
- A) Growing the tree larger
- B) Removing branches to reduce overfitting
- C) Adding more features
- D) Increasing tree depth

**Correct:** B

**Explanation:** Removing tree branches to reduce complexity and prevent overfitting.

**Chapter:** Chapter 17

---

# Question 97

**Type:** mc

What is pre-pruning?

**Options:**
- A) Pruning after tree is fully grown
- B) Stopping tree growth early based on criteria
- C) Removing the root
- D) Growing tree first then pruning

**Correct:** B

**Explanation:** Stopping tree growth early based on criteria like depth or samples.

**Chapter:** Chapter 17

---

# Question 98

**Type:** mc

What is post-pruning?

**Options:**
- A) Stopping growth early
- B) Growing full tree then removing branches
- C) Never pruning
- D) Pruning before training

**Correct:** B

**Explanation:** Growing full tree first, then removing branches based on validation performance.

**Chapter:** Chapter 17

---

# Question 99

**Type:** mc

What is the max_depth hyperparameter?

**Options:**
- A) Maximum number of features
- B) Maximum tree depth allowed
- C) Maximum number of samples
- D) Maximum accuracy

**Correct:** B

**Explanation:** Limits maximum depth of tree to prevent overfitting.

**Chapter:** Chapter 17

---

# Question 100

**Type:** mc

What is min_samples_split?

**Options:**
- A) Minimum samples in leaf
- B) Minimum samples required to split a node
- C) Minimum tree depth
- D) Minimum accuracy

**Correct:** B

**Explanation:** Minimum samples required in a node to consider splitting it further.

**Chapter:** Chapter 17

---

# Question 101

**Type:** mc

What is min_samples_leaf?

**Options:**
- A) Minimum samples to split
- B) Minimum samples required in a leaf node
- C) Minimum features
- D) Minimum depth

**Correct:** B

**Explanation:** Minimum samples that must remain in each leaf node.

**Chapter:** Chapter 17

---

# Question 102

**Type:** mc

What is min_impurity_decrease?

**Options:**
- A) Maximum impurity allowed
- B) Minimum impurity reduction required for a split
- C) Minimum number of leaves
- D) Minimum accuracy gain

**Correct:** B

**Explanation:** Minimum reduction in impurity required to perform a split.

**Chapter:** Chapter 17

---

# Question 103

**Type:** mc

What is cost-complexity pruning (CCP)?

**Options:**
- A) Pre-pruning method
- B) Post-pruning using penalty for tree complexity
- C) Feature selection
- D) Data preprocessing

**Correct:** B

**Explanation:** Post-pruning method that penalizes tree complexity using alpha parameter.

**Chapter:** Chapter 17

---

# Question 104

**Type:** mc

What is the alpha parameter in CCP?

**Options:**
- A) Learning rate
- B) Complexity penalty weight
- C) Number of trees
- D) Split criterion

**Correct:** B

**Explanation:** Alpha controls trade-off between tree complexity and training accuracy.

**Chapter:** Chapter 17

---

# Question 105

**Type:** mc

How do you choose the best alpha for CCP?

**Options:**
- A) Always use alpha=1
- B) Cross-validation to find optimal complexity
- C) Random selection
- D) Use largest alpha

**Correct:** B

**Explanation:** Use cross-validation to find alpha that minimizes validation error.

**Chapter:** Chapter 17

---

# Question 106

**Type:** mc

What is ensemble learning?

**Options:**
- A) Training one large model
- B) Combining multiple models for better predictions
- C) Using multiple features
- D) Training on multiple datasets separately

**Correct:** B

**Explanation:** Combining predictions from multiple models to improve overall performance.

**Chapter:** Chapter 18

---

# Question 107

**Type:** mc

What is bagging?

**Options:**
- A) Removing bad samples
- B) Bootstrap Aggregating - training on random samples with replacement
- C) Selecting best features
- D) Combining test results

**Correct:** B

**Explanation:** Training multiple models on random samples with replacement, then averaging predictions.

**Chapter:** Chapter 18

---

# Question 108

**Type:** mc

How does Random Forest work?

**Options:**
- A) One deep decision tree
- B) Ensemble of decision trees with random feature subsets
- C) Linear combination of features
- D) Sequential tree building

**Correct:** B

**Explanation:** Ensemble of decision trees using bootstrap samples and random feature subsets.

**Chapter:** Chapter 18

---

# Question 109

**Type:** mc

What is bootstrap sampling?

**Options:**
- A) Sampling without replacement
- B) Random sampling with replacement
- C) Taking first N samples
- D) Stratified sampling only

**Correct:** B

**Explanation:** Random sampling with replacement; same sample can be selected multiple times.

**Chapter:** Chapter 18

---

# Question 110

**Type:** mc

What is the n_estimators parameter?

**Options:**
- A) Number of features
- B) Number of trees in the ensemble
- C) Tree depth
- D) Number of samples

**Correct:** B

**Explanation:** Number of decision trees in the random forest ensemble.

**Chapter:** Chapter 18

---

# Question 111

**Type:** mc

What is max_features in Random Forest?

**Options:**
- A) Total features in dataset
- B) Number of features considered for each split
- C) Maximum tree depth
- D) Number of trees

**Correct:** B

**Explanation:** Number of features randomly selected when finding best split.

**Chapter:** Chapter 18

---

# Question 112

**Type:** mc

Why does Random Forest reduce overfitting?

**Options:**
- A) Uses smaller datasets
- B) Averaging diverse trees reduces variance
- C) Uses fewer features total
- D) Simpler than single tree

**Correct:** B

**Explanation:** Averaging many diverse trees reduces variance and overfitting.

**Chapter:** Chapter 18

---

# Question 113

**Type:** mc

What is feature importance in Random Forest?

**Options:**
- A) Number of times feature appears
- B) Average impurity decrease across all trees for each feature
- C) Feature correlation
- D) Feature mean value

**Correct:** B

**Explanation:** Measures each feature's average contribution to reducing impurity across all trees.

**Chapter:** Chapter 18

---

# Question 114

**Type:** mc

How does Random Forest handle missing values?

**Options:**
- A) Automatically in sklearn (requires imputation)
- B) Sklearn Random Forest requires preprocessing for missing values
- C) Deletes rows with missing values
- D) Replaces with zeros

**Correct:** B

**Explanation:** Sklearn requires imputation; some implementations can handle missing values natively.

**Chapter:** Chapter 18

---

# Question 115

**Type:** mc

What are out-of-bag (OOB) samples?

**Options:**
- A) Test set samples
- B) Samples not selected in bootstrap sample for a tree
- C) Outliers
- D) Validation set

**Correct:** B

**Explanation:** Samples not selected in a particular bootstrap sample; used for validation.

**Chapter:** Chapter 18

---

# Question 116

**Type:** mc

What is boosting?

**Options:**
- A) Training identical models
- B) Sequential training where each model corrects previous errors
- C) Increasing dataset size
- D) Removing weak features

**Correct:** B

**Explanation:** Sequential ensemble where each model focuses on correcting previous models' errors.

**Chapter:** Chapter 19

---

# Question 117

**Type:** mc

How does boosting differ from bagging?

**Options:**
- A) No difference
- B) Boosting is sequential, bagging is parallel
- C) Bagging is always better
- D) Boosting uses more data

**Correct:** B

**Explanation:** Boosting builds models sequentially; bagging builds all models independently in parallel.

**Chapter:** Chapter 19

---

# Question 118

**Type:** mc

What is the learning_rate parameter in boosting?

**Options:**
- A) Speed of training
- B) Weight of each tree's contribution
- C) Number of iterations
- D) Tree depth

**Correct:** B

**Explanation:** Controls how much each tree contributes; lower rate needs more trees.

**Chapter:** Chapter 19

---

# Question 119

**Type:** mc

What is the relationship between n_estimators and learning_rate?

**Options:**
- A) Independent parameters
- B) Lower learning_rate typically needs more estimators
- C) Must be equal
- D) Inversely proportional always

**Correct:** B

**Explanation:** Lower learning rate requires more estimators to achieve same performance.

**Chapter:** Chapter 19

---

# Question 120

**Type:** mc

Why does boosting often outperform bagging?

**Options:**
- A) Uses more data
- B) Focuses on hard-to-classify examples
- C) Faster training
- D) Simpler models

**Correct:** B

**Explanation:** Focuses learning on difficult cases that previous models misclassified.

**Chapter:** Chapter 19

---

# Question 121

**Type:** mc

What is the risk of boosting?

**Options:**
- A) Underfitting only
- B) Overfitting if too many iterations
- C) Cannot converge
- D) No risks

**Correct:** B

**Explanation:** Too many iterations can overfit to training data, especially with noise.

**Chapter:** Chapter 19

---

# Question 122

**Type:** mc

What is the friedman_mse criterion?

**Options:**
- A) A pruning method
- B) Mean squared error with Friedman's improvement for boosting
- C) Feature importance measure
- D) Learning rate formula

**Correct:** B

**Explanation:** Splitting criterion optimized for gradient boosting algorithms.

**Chapter:** Chapter 19

---

# Question 123

**Type:** mc

Does GradientBoostingClassifier have class_weight parameter?

**Options:**
- A) Yes, same as LogisticRegression
- B) No, it doesn't have class_weight parameter
- C) Only in newer versions
- D) Yes, but deprecated

**Correct:** B

**Explanation:** GradientBoostingClassifier doesn't have class_weight; use sample_weight in fit instead.

**Chapter:** Chapter 19

---

# Question 124

**Type:** mc

When should you use Random Forest vs Gradient Boosting?

**Options:**
- A) Always use Random Forest
- B) RF for quick baseline, GB for squeezing performance
- C) GB is always better
- D) No difference

**Correct:** B

**Explanation:** Random Forest for quick robust baseline; Gradient Boosting for maximum performance tuning.

**Chapter:** Chapter 19

---

# Question 125

**Type:** mc

What is early stopping in boosting?

**Options:**
- A) Starting late
- B) Stopping training when validation performance stops improving
- C) Training fewer trees always
- D) Pruning trees early

**Correct:** B

**Explanation:** Stops training when validation error stops decreasing to prevent overfitting.

**Chapter:** Chapter 19

---

# Question 126

**Type:** mc

How do you import NumPy with standard alias?

**Options:**
- A) import numpy
- B) import numpy as np
- C) from numpy import all
- D) import np

**Correct:** B

**Explanation:** Standard convention: import numpy as np for brevity.

**Chapter:** All chapters (imports)

---

# Question 127

**Type:** mc

How do you create a NumPy array from a list?

**Options:**
- A) np.list([1,2,3])
- B) np.array([1,2,3])
- C) np.create([1,2,3])
- D) numpy.list([1,2,3])

**Correct:** B

**Explanation:** np.array() converts Python lists to NumPy arrays.

**Chapter:** Chapter 1

---

# Question 128

**Type:** mc

How do you create an array of zeros with shape (5, 3)?

**Options:**
- A) np.zeros(5,3)
- B) np.zeros((5, 3))
- C) np.zero((5, 3))
- D) np.zeros[5,3]

**Correct:** B

**Explanation:** Use tuple (5, 3) to specify shape with 5 rows, 3 columns.

**Chapter:** Data preprocessing

---

# Question 129

**Type:** mc

How do you create an array of ones?

**Options:**
- A) np.one((2,3))
- B) np.ones((2,3))
- C) np.ones(2,3)
- D) np.create_ones((2,3))

**Correct:** B

**Explanation:** np.ones() creates array filled with ones in specified shape.

**Chapter:** Data preprocessing

---

# Question 130

**Type:** mc

How do you create a range of numbers from 0 to 10?

**Options:**
- A) np.range(0, 11)
- B) np.arange(11)
- C) np.sequence(0, 10)
- D) np.array(range(10))

**Correct:** B

**Explanation:** np.arange(11) creates array [0,1,2,...,10]; 11 is exclusive upper bound.

**Chapter:** Visualization chapters

---

# Question 131

**Type:** mc

How do you reshape a 1D array to 2D?

**Options:**
- A) arr.shape((5,2))
- B) arr.reshape((5, 2))
- C) arr.resize((5, 2))
- D) arr.change_shape((5,2))

**Correct:** B

**Explanation:** reshape() changes array dimensions; requires compatible total number of elements.

**Chapter:** Model fitting chapters

---

# Question 132

**Type:** mc

What does .reshape(-1, 1) do?

**Options:**
- A) Creates a 2D array with 1 row
- B) Converts to column vector (2D with 1 column)
- C) Flattens the array
- D) Removes dimension

**Correct:** B

**Explanation:** Creates column vector: -1 infers rows, 1 specifies single column.

**Chapter:** Model input preparation

---

# Question 133

**Type:** mc

How do you calculate the mean of an array?

**Options:**
- A) arr.average()
- B) arr.mean() or np.mean(arr)
- C) arr.avg()
- D) np.average(arr) only

**Correct:** B

**Explanation:** Both arr.mean() and np.mean(arr) calculate the arithmetic mean.

**Chapter:** Chapter 1 (probability)

---

# Question 134

**Type:** mc

How do you calculate the sum of an array?

**Options:**
- A) arr.total()
- B) arr.sum() or np.sum(arr)
- C) arr.add()
- D) np.total(arr)

**Correct:** B

**Explanation:** Both arr.sum() and np.sum(arr) calculate the total sum.

**Chapter:** Metric calculations

---

# Question 135

**Type:** mc

How do you find the maximum value in an array?

**Options:**
- A) arr.maximum()
- B) arr.max() or np.max(arr)
- C) arr.largest()
- D) np.maximum(arr)

**Correct:** B

**Explanation:** Both arr.max() and np.max(arr) return the maximum value.

**Chapter:** Data exploration

---

# Question 136

**Type:** mc

How do you create a boolean mask where values > 5?

**Options:**
- A) arr.filter(> 5)
- B) arr > 5
- C) arr.greater(5)
- D) np.where(arr, 5)

**Correct:** B

**Explanation:** Comparison operators create boolean arrays for filtering or counting.

**Chapter:** Data filtering

---

# Question 137

**Type:** mc

How do you count True values in a boolean array?

**Options:**
- A) bool_arr.count()
- B) np.sum(bool_arr) or bool_arr.sum()
- C) bool_arr.total()
- D) len(bool_arr)

**Correct:** B

**Explanation:** Sum treats True as 1 and False as 0, counting True values.

**Chapter:** Metric calculations

---

# Question 138

**Type:** mc

How do you use np.where() for conditional selection?

**Options:**
- A) np.where(arr > 0)
- B) np.where(condition, value_if_true, value_if_false)
- C) np.where(arr, value)
- D) np.select(condition)

**Correct:** B

**Explanation:** np.where(condition, x, y) returns x where condition True, else y.

**Chapter:** Chapter 8 (threshold adjustment)

---

# Question 139

**Type:** mc

How do you combine boolean arrays with &, |?

**Options:**
- A) arr1 and arr2, arr1 or arr2
- B) (arr1 > 0) & (arr2 < 10)
- C) arr1 && arr2, arr1 || arr2
- D) np.and(arr1, arr2)

**Correct:** B

**Explanation:** Use & for AND, | for OR; parentheses required for compound conditions.

**Chapter:** Data filtering

---

# Question 140

**Type:** mc

What does np.any() do?

**Options:**
- A) Returns all True values
- B) Returns True if any element is True
- C) Returns first True value
- D) Counts True values

**Correct:** B

**Explanation:** Returns True if at least one element in array is True.

**Chapter:** Data validation

---

# Question 141

**Type:** mc

How do you import pandas with standard alias?

**Options:**
- A) import pandas
- B) import pandas as pd
- C) from pandas import all
- D) import pd

**Correct:** B

**Explanation:** Standard convention: import pandas as pd for brevity.

**Chapter:** All chapters (imports)

---

# Question 142

**Type:** mc

How do you read a CSV file from a URL?

**Options:**
- A) pd.read(url)
- B) pd.read_csv(url)
- C) pd.load_csv(url)
- D) pd.from_csv(url)

**Correct:** B

**Explanation:** pd.read_csv() works with URLs and local file paths.

**Chapter:** All dataset loading

---

# Question 143

**Type:** mc

How do you display first 5 rows of a DataFrame?

**Options:**
- A) df.first(5)
- B) df.head()
- C) df.show(5)
- D) df.top(5)

**Correct:** B

**Explanation:** df.head() shows first 5 rows by default; optional argument for different number.

**Chapter:** Data exploration

---

# Question 144

**Type:** mc

What does .info() show?

**Options:**
- A) Only data types
- B) Summary: columns, dtypes, non-null counts, memory
- C) Only first few rows
- D) Statistical summary

**Correct:** B

**Explanation:** Shows column names, data types, non-null counts, and memory usage.

**Chapter:** Data exploration

---

# Question 145

**Type:** mc

What does .describe() show?

**Options:**
- A) Data types
- B) Statistical summary (count, mean, std, min, max, quartiles)
- C) Column names
- D) Missing values

**Correct:** B

**Explanation:** Shows count, mean, standard deviation, min, quartiles, and max for numeric columns.

**Chapter:** Data exploration

---

# Question 146

**Type:** mc

How do you select a single column?

**Options:**
- A) df.column_name or df['column_name']
- B) df.get('column_name')
- C) df.select('column_name')
- D) df:column_name

**Correct:** A

**Explanation:** Both df.column_name and df['column_name'] select a single column Series.

**Chapter:** Feature selection

---

# Question 147

**Type:** mc

How do you select multiple columns?

**Options:**
- A) df.col1, col2
- B) df[['col1', 'col2']]
- C) df['col1', 'col2']
- D) df.get(['col1', 'col2'])

**Correct:** B

**Explanation:** Double brackets return DataFrame with selected columns.

**Chapter:** Feature selection

---

# Question 148

**Type:** mc

How do you select rows using iloc?

**Options:**
- A) df.iloc['row_name']
- B) df.iloc[0:5] or df.iloc[0]
- C) df.iloc('row_name')
- D) df.iloc{'row': 0}

**Correct:** B

**Explanation:** iloc uses integer positions; row 0 is first, row 5 is sixth.

**Chapter:** Data subsetting

---

# Question 149

**Type:** mc

How do you select rows using loc?

**Options:**
- A) df.loc[0]
- B) df.loc[row_label] or df.loc[condition]
- C) df.loc(row_label)
- D) df.loc{row_label}

**Correct:** B

**Explanation:** loc uses labels; can use boolean conditions for filtering.

**Chapter:** Data subsetting

---

# Question 150

**Type:** mc

How do you filter rows based on a condition?

**Options:**
- A) df.filter(df['col'] > 5)
- B) df[df['col'] > 5]
- C) df.where(df['col'] > 5)
- D) df.select(df['col'] > 5)

**Correct:** B

**Explanation:** Boolean indexing: condition in brackets returns filtered DataFrame.

**Chapter:** Data exploration

---

# Question 151

**Type:** mc

How do you create dummy variables with pd.get_dummies()?

**Options:**
- A) pd.dummy(df, columns=['col'])
- B) pd.get_dummies(df, columns=['col'])
- C) df.get_dummies(columns=['col'])
- D) pd.create_dummies(df, 'col')

**Correct:** B

**Explanation:** Creates dummy variables for specified categorical columns.

**Chapter:** All preprocessing

---

# Question 152

**Type:** mc

What does drop_first=True do in get_dummies()?

**Options:**
- A) Drops first row
- B) Drops first dummy column to avoid multicollinearity
- C) Drops first feature
- D) Drops missing values

**Correct:** B

**Explanation:** Drops one dummy column per category to avoid multicollinearity.

**Chapter:** Categorical encoding

---

# Question 153

**Type:** mc

How do you drop columns from a DataFrame?

**Options:**
- A) df.remove(['col'])
- B) df.drop(['col'], axis=1) or df.drop(columns=['col'])
- C) df.delete(['col'])
- D) df.drop_column(['col'])

**Correct:** B

**Explanation:** axis=1 means drop columns; axis=0 would drop rows.

**Chapter:** Feature selection

---

# Question 154

**Type:** mc

What does inplace=True mean?

**Options:**
- A) Creates a copy
- B) Modifies DataFrame in place, returns None
- C) Returns new DataFrame
- D) Validates the operation

**Correct:** B

**Explanation:** Modifies original DataFrame instead of returning new copy.

**Chapter:** Data manipulation

---

# Question 155

**Type:** mc

How do you check for missing values?

**Options:**
- A) df.missing()
- B) df.isna() or df.isnull()
- C) df.check_null()
- D) df.find_missing()

**Correct:** B

**Explanation:** Both methods check for missing values; isna() is newer alias.

**Chapter:** Data quality checks

---

# Question 156

**Type:** mc

What does .value_counts() do?

**Options:**
- A) Counts total values
- B) Returns frequency of unique values in a Series
- C) Counts missing values
- D) Counts all values

**Correct:** B

**Explanation:** Counts frequency of each unique value in descending order.

**Chapter:** Class distribution checks

---

# Question 157

**Type:** mc

How do you get proportions instead of counts?

**Options:**
- A) value_counts(proportion=True)
- B) value_counts(normalize=True)
- C) value_counts(percent=True)
- D) value_counts()/len(df)

**Correct:** B

**Explanation:** normalize=True converts counts to proportions summing to 1.

**Chapter:** Probability calculations

---

# Question 158

**Type:** mc

How do you sort value_counts() by index?

**Options:**
- A) value_counts(sort=True)
- B) value_counts().sort_index()
- C) value_counts().sort()
- D) value_counts().order()

**Correct:** B

**Explanation:** sort_index() sorts by the index (values) instead of counts.

**Chapter:** Ordered data exploration

---

# Question 159

**Type:** mc

What does .unique() return?

**Options:**
- A) Count of unique values
- B) Array of unique values
- C) Boolean mask
- D) Frequency of values

**Correct:** B

**Explanation:** Returns array of unique values without frequencies.

**Chapter:** Category identification

---

# Question 160

**Type:** mc

How do you group by a column and calculate mean?

**Options:**
- A) df.group('col').mean()
- B) df.groupby('col').mean()
- C) df.groupby('col', mean=True)
- D) df.mean(groupby='col')

**Correct:** B

**Explanation:** Groups by column, then calculates mean for each group.

**Chapter:** Group analysis

---

# Question 161

**Type:** mc

How do you import matplotlib.pyplot with standard alias?

**Options:**
- A) import matplotlib
- B) import matplotlib.pyplot as plt
- C) from matplotlib import pyplot
- D) import plt

**Correct:** B

**Explanation:** Standard convention: import matplotlib.pyplot as plt for brevity.

**Chapter:** All chapters (imports)

---

# Question 162

**Type:** mc

How do you create a simple line plot?

**Options:**
- A) plt.line(x, y)
- B) plt.plot(x, y)
- C) plt.draw(x, y)
- D) plt.line_plot(x, y)

**Correct:** B

**Explanation:** plt.plot() creates line plot connecting points.

**Chapter:** Visualization chapters

---

# Question 163

**Type:** mc

How do you create a scatter plot?

**Options:**
- A) plt.plot(x, y, 'o')
- B) plt.scatter(x, y)
- C) plt.points(x, y)
- D) plt.dot(x, y)

**Correct:** B

**Explanation:** plt.scatter() creates scatter plot with individual points.

**Chapter:** Data exploration

---

# Question 164

**Type:** mc

How do you create a bar plot?

**Options:**
- A) plt.bars(x, y)
- B) plt.bar(x, height)
- C) plt.barplot(x, y)
- D) plt.column(x, y)

**Correct:** B

**Explanation:** plt.bar() creates bar chart; height parameter specifies bar heights.

**Chapter:** Distribution visualization

---

# Question 165

**Type:** mc

How do you add labels to x and y axes?

**Options:**
- A) plt.labels('x', 'y')
- B) plt.xlabel('x'); plt.ylabel('y')
- C) plt.set_labels('x', 'y')
- D) plt.axis_labels('x', 'y')

**Correct:** B

**Explanation:** xlabel() and ylabel() add descriptive labels to axes.

**Chapter:** All visualizations

---

# Question 166

**Type:** mc

How do you set figure size?

**Options:**
- A) plt.size(10, 6)
- B) plt.figure(figsize=(10, 6))
- C) plt.set_size(10, 6)
- D) plt.figsize(10, 6)

**Correct:** B

**Explanation:** figsize tuple (width, height) in inches sets figure size.

**Chapter:** All visualizations

---

# Question 167

**Type:** mc

How do you add a title to a plot?

**Options:**
- A) plt.header('Title')
- B) plt.title('Title')
- C) plt.set_title('Title')
- D) plt.name('Title')

**Correct:** B

**Explanation:** plt.title() adds title text above the plot.

**Chapter:** All visualizations

---

# Question 168

**Type:** mc

How do you add a legend?

**Options:**
- A) plt.show_legend()
- B) plt.legend()
- C) plt.add_legend()
- D) plt.key()

**Correct:** B

**Explanation:** plt.legend() displays legend box showing labels from plot commands.

**Chapter:** Multi-series plots

---

# Question 169

**Type:** mc

How do you add a grid?

**Options:**
- A) plt.show_grid()
- B) plt.grid(True) or plt.grid()
- C) plt.add_grid()
- D) plt.gridlines()

**Correct:** B

**Explanation:** plt.grid() or plt.grid(True) adds gridlines to plot.

**Chapter:** All visualizations

---

# Question 170

**Type:** mc

How do you save a figure to file?

**Options:**
- A) plt.save('file.png')
- B) plt.savefig('file.png')
- C) plt.export('file.png')
- D) plt.write('file.png')

**Correct:** B

**Explanation:** plt.savefig() saves current figure to file; supports PNG, PDF, SVG formats.

**Chapter:** Report generation

---

# Question 171

**Type:** mc

How do you create subplots (2 rows, 2 cols)?

**Options:**
- A) plt.subplot(2, 2)
- B) fig, axes = plt.subplots(2, 2)
- C) plt.subplots((2, 2))
- D) plt.create_subplots(2, 2)

**Correct:** B

**Explanation:** Returns figure and array of axes objects for subplots.

**Chapter:** Multi-panel visualizations

---

# Question 172

**Type:** mc

How do you access a specific subplot in axes array?

**Options:**
- A) axes.get(0, 0)
- B) axes[0, 0] or axes[0][0]
- C) axes(0, 0)
- D) axes.subplot(0, 0)

**Correct:** B

**Explanation:** Indexing axes array accesses individual subplot for plotting.

**Chapter:** Multi-panel visualizations

---

# Question 173

**Type:** mc

How do you adjust spacing between subplots?

**Options:**
- A) plt.spacing()
- B) plt.tight_layout() or plt.subplots_adjust()
- C) plt.adjust_space()
- D) plt.set_spacing()

**Correct:** B

**Explanation:** tight_layout() auto-adjusts; subplots_adjust() for manual spacing control.

**Chapter:** Layout management

---

# Question 174

**Type:** mc

How do you create horizontal and vertical lines?

**Options:**
- A) plt.line(x, 'h')
- B) plt.axhline(y), plt.axvline(x)
- C) plt.hline(y), plt.vline(x)
- D) plt.draw_line(x, y)

**Correct:** B

**Explanation:** axhline draws horizontal line at y; axvline draws vertical line at x.

**Chapter:** Reference lines

---

# Question 175

**Type:** mc

How do you set axis limits?

**Options:**
- A) plt.limits(0, 10)
- B) plt.xlim(0, 10); plt.ylim(0, 10)
- C) plt.set_limits(0, 10)
- D) plt.axis_limits(0, 10)

**Correct:** B

**Explanation:** xlim() and ylim() set axis ranges for better visualization.

**Chapter:** Plot customization

---

# Question 176

**Type:** mc

How do you import train_test_split?

**Options:**
- A) from sklearn import train_test_split
- B) from sklearn.model_selection import train_test_split
- C) from sklearn.split import train_test_split
- D) import sklearn.train_test_split

**Correct:** B

**Explanation:** Import train_test_split from sklearn.model_selection submodule.

**Chapter:** All chapters with modeling

---

# Question 177

**Type:** mc

What are the main parameters of train_test_split()?

**Options:**
- A) X only
- B) X, y, test_size, random_state, stratify
- C) data, labels
- D) features, target, split

**Correct:** B

**Explanation:** Key parameters: X, y, test_size, random_state, stratify for balanced splits.

**Chapter:** Data splitting

---

# Question 178

**Type:** mc

What does test_size=0.2 mean?

**Options:**
- A) 2% for testing
- B) 20% for testing, 80% for training
- C) 0.2 samples for testing
- D) 20 samples for testing

**Correct:** B

**Explanation:** 20% of data for testing, 80% for training.

**Chapter:** Data splitting

---

# Question 179

**Type:** mc

What does random_state do?

**Options:**
- A) Increases randomness
- B) Sets seed for reproducible random splits
- C) Randomizes features
- D) Shuffles data

**Correct:** B

**Explanation:** Sets random seed for reproducible splits across multiple runs.

**Chapter:** Reproducibility

---

# Question 180

**Type:** mc

What does stratify parameter do?

**Options:**
- A) Sorts the data
- B) Preserves class distribution in train/test splits
- C) Normalizes features
- D) Removes outliers

**Correct:** B

**Explanation:** Maintains class distribution proportions in both train and test sets.

**Chapter:** Imbalanced data handling

---

# Question 181

**Type:** mc

How do you import KFold?

**Options:**
- A) from sklearn import KFold
- B) from sklearn.model_selection import KFold
- C) from sklearn.cross_validation import KFold
- D) import sklearn.KFold

**Correct:** B

**Explanation:** Import KFold from sklearn.model_selection for cross-validation.

**Chapter:** Chapter 7

---

# Question 182

**Type:** mc

What parameters does KFold require?

**Options:**
- A) data, labels
- B) n_splits, shuffle, random_state
- C) X, y, splits
- D) folds only

**Correct:** B

**Explanation:** n_splits defines folds; shuffle randomizes; random_state ensures reproducibility.

**Chapter:** Cross-validation

---

# Question 183

**Type:** mc

What does n_splits parameter control?

**Options:**
- A) Number of features
- B) Number of folds in cross-validation
- C) Train/test ratio
- D) Number of models

**Correct:** B

**Explanation:** n_splits determines how many folds to split data into.

**Chapter:** CV configuration

---

# Question 184

**Type:** mc

What does shuffle=True do in KFold?

**Options:**
- A) Sorts data
- B) Randomly shuffles data before splitting into folds
- C) Removes randomness
- D) Stratifies classes

**Correct:** B

**Explanation:** Randomly shuffles data before splitting into folds for better generalization.

**Chapter:** CV configuration

---

# Question 185

**Type:** mc

Why use cross-validation?

**Options:**
- A) Faster training
- B) More robust performance estimates, uses all data for train/test
- C) Increases accuracy
- D) Reduces dataset size

**Correct:** B

**Explanation:** Uses all data for both training and testing, more robust than single split.

**Chapter:** Model evaluation

---

# Question 186

**Type:** mc

How do you import GridSearchCV?

**Options:**
- A) from sklearn import GridSearchCV
- B) from sklearn.model_selection import GridSearchCV
- C) from sklearn.grid_search import GridSearchCV
- D) import sklearn.GridSearchCV

**Correct:** B

**Explanation:** Import GridSearchCV from sklearn.model_selection for hyperparameter tuning.

**Chapter:** Chapter 7

---

# Question 187

**Type:** mc

What is the param_grid parameter?

**Options:**
- A) Grid of data points
- B) Dictionary of hyperparameters to search
- C) Feature grid
- D) Cross-validation folds

**Correct:** B

**Explanation:** Dictionary mapping parameter names to lists of values to try.

**Chapter:** Hyperparameter tuning

---

# Question 188

**Type:** mc

How do you specify parameter names for pipeline steps?

**Options:**
- A) stepname.param
- B) stepname__param (double underscore)
- C) stepname_param
- D) stepname:param

**Correct:** B

**Explanation:** Use double underscore: 'stepname__param' to access nested pipeline parameters.

**Chapter:** Pipeline tuning

---

# Question 189

**Type:** mc

What does the scoring parameter do?

**Options:**
- A) Scores the data
- B) Specifies metric to optimize (e.g., 'accuracy', 'recall')
- C) Normalizes scores
- D) Counts correct predictions

**Correct:** B

**Explanation:** Specifies which metric to optimize during hyperparameter search.

**Chapter:** Metric selection

---

# Question 190

**Type:** mc

How do you access best parameters after fitting?

**Options:**
- A) grid.parameters
- B) grid.best_params_
- C) grid.get_best_params()
- D) grid.params

**Correct:** B

**Explanation:** Access best_params_ attribute to get optimal hyperparameter values.

**Chapter:** Result retrieval

---

# Question 191

**Type:** mc

How do you access the best estimator?

**Options:**
- A) grid.model
- B) grid.best_estimator_
- C) grid.get_estimator()
- D) grid.estimator

**Correct:** B

**Explanation:** Access best_estimator_ attribute to get model fitted with best parameters.

**Chapter:** Model retrieval

---

# Question 192

**Type:** mc

What does fit() do on GridSearchCV object?

**Options:**
- A) Only fits one model
- B) Fits all parameter combinations with CV, selects best
- C) Validates parameters
- D) Transforms data

**Correct:** B

**Explanation:** Tries all parameter combinations using cross-validation, stores best model.

**Chapter:** Model training

---

# Question 193

**Type:** mc

What is n_jobs parameter?

**Options:**
- A) Number of models
- B) Number of parallel jobs/CPU cores to use
- C) Number of iterations
- D) Job priority

**Correct:** B

**Explanation:** Number of parallel jobs; -1 uses all CPU cores.

**Chapter:** Parallel processing

---

# Question 194

**Type:** mc

What does n_jobs=-1 mean?

**Options:**
- A) No parallelization
- B) Use all available CPU cores
- C) Use 1 core
- D) Invalid value

**Correct:** B

**Explanation:** Uses all available CPU cores for parallel cross-validation.

**Chapter:** Resource utilization

---

# Question 195

**Type:** mc

How do you specify custom scoring function?

**Options:**
- A) scoring=my_function
- B) scoring=make_scorer(my_function)
- C) scoring='custom'
- D) custom_scoring=my_function

**Correct:** B

**Explanation:** make_scorer() wraps scoring function for use in GridSearchCV.

**Chapter:** Chapter 8

---

# Question 196

**Type:** mc

How do you import confusion_matrix?

**Options:**
- A) from sklearn import confusion_matrix
- B) from sklearn.metrics import confusion_matrix
- C) from sklearn.evaluation import confusion_matrix
- D) import sklearn.confusion_matrix

**Correct:** B

**Explanation:** Import confusion_matrix from sklearn.metrics for evaluation.

**Chapter:** Chapter 6

---

# Question 197

**Type:** mc

What are the arguments to confusion_matrix()?

**Options:**
- A) predictions, actual
- B) y_true, y_pred
- C) actual, predicted
- D) y_pred, y_true

**Correct:** B

**Explanation:** First argument: true labels; second argument: predicted labels.

**Chapter:** Evaluation

---

# Question 198

**Type:** mc

What is the shape of a binary confusion matrix?

**Options:**
- A) (1, 2)
- B) (2, 2)
- C) (2, 1)
- D) (4, 4)

**Correct:** B

**Explanation:** 2x2 matrix for binary classification: rows actual, columns predicted.

**Chapter:** Chapter 6

---

# Question 199

**Type:** mc

What is the shape of a 3-class confusion matrix?

**Options:**
- A) (2, 2)
- B) (3, 3)
- C) (3, 2)
- D) (9, 9)

**Correct:** B

**Explanation:** 3x3 matrix for 3-class problem: K×K for K classes.

**Chapter:** Multiclass evaluation

---

# Question 200

**Type:** mc

How do you extract TP, TN, FP, FN from confusion matrix?

**Options:**
- A) cm.get_values()
- B) TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1]
- C) cm.extract()
- D) cm.values()

**Correct:** B

**Explanation:** Position [0,0]=TN, [0,1]=FP, [1,0]=FN, [1,1]=TP for binary.

**Chapter:** Manual metric calculation

---

# Question 201

**Type:** mc

How do you import classification_report?

**Options:**
- A) from sklearn import classification_report
- B) from sklearn.metrics import classification_report
- C) from sklearn.evaluation import classification_report
- D) import sklearn.classification_report

**Correct:** B

**Explanation:** Import classification_report from sklearn.metrics for detailed metrics.

**Chapter:** Chapter 6

---

# Question 202

**Type:** mc

What metrics does classification_report show?

**Options:**
- A) Only accuracy
- B) Precision, recall, f1-score, support per class
- C) Only confusion matrix
- D) Only AUC

**Correct:** B

**Explanation:** Shows precision, recall, f1-score, and support for each class.

**Chapter:** Comprehensive evaluation

---

# Question 203

**Type:** mc

How should you display classification_report output?

**Options:**
- A) Just call it
- B) print(classification_report(y_true, y_pred))
- C) classification_report.show()
- D) display(classification_report())

**Correct:** B

**Explanation:** Use print() to display formatted text report.

**Chapter:** Output formatting (with print)

---

# Question 204

**Type:** mc

What is support in classification report?

**Options:**
- A) Model confidence
- B) Number of actual occurrences of each class
- C) Prediction accuracy
- D) Feature importance

**Correct:** B

**Explanation:** Support is the number of actual samples for each class.

**Chapter:** Metric interpretation

---

# Question 205

**Type:** mc

What is macro avg vs weighted avg?

**Options:**
- A) They are the same
- B) Macro: unweighted mean; Weighted: weighted by support
- C) Macro is better
- D) Weighted ignores some classes

**Correct:** B

**Explanation:** Macro: simple average; weighted: average weighted by class sample counts.

**Chapter:** Multiclass metrics

---

# Question 206

**Type:** mc

How do you import accuracy_score?

**Options:**
- A) from sklearn import accuracy_score
- B) from sklearn.metrics import accuracy_score
- C) from sklearn.scores import accuracy_score
- D) import sklearn.accuracy

**Correct:** B

**Explanation:** Import accuracy_score from sklearn.metrics for accuracy calculation.

**Chapter:** Basic metrics

---

# Question 207

**Type:** mc

How do you import precision_score?

**Options:**
- A) from sklearn import precision_score
- B) from sklearn.metrics import precision_score
- C) from sklearn.scores import precision_score
- D) import sklearn.precision

**Correct:** B

**Explanation:** Import precision_score from sklearn.metrics for precision calculation.

**Chapter:** Precision calculation

---

# Question 208

**Type:** mc

How do you import recall_score?

**Options:**
- A) from sklearn import recall_score
- B) from sklearn.metrics import recall_score
- C) from sklearn.scores import recall_score
- D) import sklearn.recall

**Correct:** B

**Explanation:** Import recall_score from sklearn.metrics for recall calculation.

**Chapter:** Recall calculation

---

# Question 209

**Type:** mc

How do you import f1_score?

**Options:**
- A) from sklearn import f1_score
- B) from sklearn.metrics import f1_score
- C) from sklearn.scores import f1_score
- D) import sklearn.f1

**Correct:** B

**Explanation:** Import f1_score from sklearn.metrics for F1-score calculation.

**Chapter:** F1 calculation

---

# Question 210

**Type:** mc

What is the pos_label parameter?

**Options:**
- A) Position of label
- B) Specifies which class is 'positive' in binary classification
- C) Label value
- D) Positive score

**Correct:** B

**Explanation:** Specifies which class is positive in binary classification metrics.

**Chapter:** Binary classification

---

# Question 211

**Type:** mc

How do you import make_scorer?

**Options:**
- A) from sklearn import make_scorer
- B) from sklearn.metrics import make_scorer
- C) from sklearn.scoring import make_scorer
- D) import sklearn.make_scorer

**Correct:** B

**Explanation:** Import make_scorer from sklearn.metrics to create custom scorers.

**Chapter:** Chapter 7

---

# Question 212

**Type:** mc

How do you create a custom recall scorer?

**Options:**
- A) scorer = recall_score
- B) scorer = make_scorer(recall_score, pos_label='yes')
- C) scorer = custom_scorer(recall)
- D) scorer = create_scorer('recall')

**Correct:** B

**Explanation:** Wrap recall_score with make_scorer, specify pos_label for binary classification.

**Chapter:** Custom metrics

---

# Question 213

**Type:** mc

What parameters does make_scorer accept?

**Options:**
- A) Only the scoring function
- B) score_func, greater_is_better, needs_proba, **kwargs
- C) function, name
- D) metric, data

**Correct:** B

**Explanation:** Takes scoring function, greater_is_better flag, and additional function parameters.

**Chapter:** Scorer configuration

---

# Question 214

**Type:** mc

Why use make_scorer with GridSearchCV?

**Options:**
- A) Required for GridSearchCV
- B) To use custom metrics or specify metric parameters
- C) To improve accuracy
- D) To speed up search

**Correct:** B

**Explanation:** Enables custom metrics and metric parameters in hyperparameter search.

**Chapter:** Custom optimization

---

# Question 215

**Type:** mc

How do you specify pos_label in make_scorer?

**Options:**
- A) make_scorer(func, positive='yes')
- B) make_scorer(func, pos_label='yes')
- C) make_scorer(func, label='yes')
- D) make_scorer(func).set_label('yes')

**Correct:** B

**Explanation:** Pass pos_label as keyword argument to make_scorer.

**Chapter:** Binary classification

---

# Question 216

**Type:** mc

How do you import StandardScaler?

**Options:**
- A) from sklearn import StandardScaler
- B) from sklearn.preprocessing import StandardScaler
- C) from sklearn.scaler import StandardScaler
- D) import sklearn.StandardScaler

**Correct:** B

**Explanation:** Import StandardScaler from sklearn.preprocessing for feature scaling.

**Chapter:** Chapter 9

---

# Question 217

**Type:** mc

What does StandardScaler do?

**Options:**
- A) Scales to [0,1]
- B) Standardizes features: mean=0, std=1
- C) Normalizes to unit length
- D) Removes outliers

**Correct:** B

**Explanation:** Subtracts mean and divides by standard deviation: z-score normalization.

**Chapter:** Feature scaling

---

# Question 218

**Type:** mc

How do you fit and transform data?

**Options:**
- A) scaler.transform(X_train)
- B) scaler.fit(X_train); X_train_scaled = scaler.transform(X_train)
- C) scaler.scale(X_train)
- D) scaler(X_train)

**Correct:** B

**Explanation:** fit() learns parameters from training data; transform() applies transformation.

**Chapter:** Preprocessing workflow

---

# Question 219

**Type:** mc

What is the difference between fit(), transform(), and fit_transform()?

**Options:**
- A) No difference
- B) fit: learns parameters; transform: applies; fit_transform: both
- C) All do the same thing
- D) fit_transform is deprecated

**Correct:** B

**Explanation:** fit learns mean/std; transform applies; fit_transform does both together.

**Chapter:** Scaler usage

---

# Question 220

**Type:** mc

Should you fit scaler on test data?

**Options:**
- A) Yes, fit separately on test
- B) No, only fit on training, transform test
- C) Fit on both together
- D) Doesn't matter

**Correct:** B

**Explanation:** No, fit only on training to prevent data leakage from test set.

**Chapter:** Data leakage prevention

---

# Question 221

**Type:** mc

How do you import make_pipeline?

**Options:**
- A) from sklearn import make_pipeline
- B) from sklearn.pipeline import make_pipeline
- C) from sklearn.preprocessing import make_pipeline
- D) import sklearn.make_pipeline

**Correct:** B

**Explanation:** Import make_pipeline from sklearn.pipeline to chain transformers and estimators.

**Chapter:** Chapter 9

---

# Question 222

**Type:** mc

What is a pipeline?

**Options:**
- A) A data structure
- B) Chains preprocessing and model steps sequentially
- C) A plotting tool
- D) A data loader

**Correct:** B

**Explanation:** Chains preprocessing and model steps ensuring consistent transformations.

**Chapter:** Workflow organization

---

# Question 223

**Type:** mc

How do you create a pipeline with scaler and model?

**Options:**
- A) pipeline([scaler, model])
- B) make_pipeline(StandardScaler(), LogisticRegression())
- C) Pipeline(scaler, model)
- D) create_pipeline(scaler, model)

**Correct:** B

**Explanation:** Pass steps in order: make_pipeline(StandardScaler(), LogisticRegression()).

**Chapter:** Standard workflow

---

# Question 224

**Type:** mc

How do you access steps in a pipeline?

**Options:**
- A) pipeline.get_steps()
- B) pipeline.steps or pipeline.named_steps
- C) pipeline[0]
- D) pipeline.get(0)

**Correct:** B

**Explanation:** Use .steps attribute or .named_steps dictionary to access pipeline components.

**Chapter:** Pipeline inspection

---

# Question 225

**Type:** mc

What is named_steps?

**Options:**
- A) Step names only
- B) Dictionary to access pipeline steps by name
- C) Numbered steps
- D) Step parameters

**Correct:** B

**Explanation:** Dictionary mapping step names to transformer/estimator objects.

**Chapter:** Step access

---

# Question 226

**Type:** mc

How do you create dummy variables in pandas?

**Options:**
- A) df.dummies()
- B) pd.get_dummies(df)
- C) df.create_dummies()
- D) pd.dummy(df)

**Correct:** B

**Explanation:** pd.get_dummies() creates binary dummy variables for categorical features.

**Chapter:** All preprocessing

---

# Question 227

**Type:** mc

What does drop_first=True prevent?

**Options:**
- A) Data loss
- B) Multicollinearity (dummy variable trap)
- C) Missing values
- D) Overfitting

**Correct:** B

**Explanation:** Prevents multicollinearity by avoiding dummy variable trap.

**Chapter:** Multicollinearity

---

# Question 228

**Type:** mc

How do you handle multiple categorical columns?

**Options:**
- A) One at a time only
- B) pd.get_dummies(df, columns=['col1', 'col2'])
- C) Cannot handle multiple
- D) Use separate function

**Correct:** B

**Explanation:** Specify all categorical columns in columns parameter as list.

**Chapter:** Multi-column encoding

---

# Question 229

**Type:** mc

What is the columns parameter in get_dummies()?

**Options:**
- A) Columns to keep
- B) Specific columns to encode
- C) Output column names
- D) Columns to drop

**Correct:** B

**Explanation:** List of categorical column names to encode; others remain unchanged.

**Chapter:** Selective encoding

---

# Question 230

**Type:** mc

How do you drop specific columns after get_dummies?

**Options:**
- A) df.remove(columns)
- B) df.drop(columns=['col'], axis=1)
- C) df.delete(['col'])
- D) df.drop_columns(['col'])

**Correct:** B

**Explanation:** Use drop() with axis=1 or columns parameter to remove unwanted columns.

**Chapter:** Feature selection

---

# Question 231

**Type:** mc

How do you import LogisticRegression?

**Options:**
- A) from sklearn import LogisticRegression
- B) from sklearn.linear_model import LogisticRegression
- C) from sklearn.models import LogisticRegression
- D) import sklearn.LogisticRegression

**Correct:** B

**Explanation:** Import LogisticRegression from sklearn.linear_model for classification.

**Chapter:** All LR chapters

---

# Question 232

**Type:** mc

What is the C parameter?

**Options:**
- A) Number of classes
- B) Inverse of regularization strength
- C) Convergence criterion
- D) Class weight

**Correct:** B

**Explanation:** C is inverse of regularization strength; smaller C means more regularization.

**Chapter:** Chapter 7

---

# Question 233

**Type:** mc

What is the class_weight parameter?

**Options:**
- A) Weight of classifier
- B) Weights for classes (e.g., 'balanced')
- C) Feature weights
- D) Sample weights

**Correct:** B

**Explanation:** Sets weights for classes to handle imbalanced datasets.

**Chapter:** Chapter 8

---

# Question 234

**Type:** mc

What does multi_class='multinomial' do?

**Options:**
- A) Handles binary only
- B) Uses multinomial loss for multiclass
- C) Creates multiple models
- D) Not a valid parameter

**Correct:** B

**Explanation:** Uses multinomial loss with softmax for true multiclass classification.

**Chapter:** Chapter 14

---

# Question 235

**Type:** mc

How do you access coefficients after fitting?

**Options:**
- A) model.weights
- B) model.coef_
- C) model.coefficients
- D) model.get_coef()

**Correct:** B

**Explanation:** Access coef_ attribute after fitting for coefficient values.

**Chapter:** Chapter 5

---

# Question 236

**Type:** mc

How do you access intercept after fitting?

**Options:**
- A) model.bias
- B) model.intercept_
- C) model.get_intercept()
- D) model.beta0

**Correct:** B

**Explanation:** Access intercept_ attribute after fitting for intercept value.

**Chapter:** Chapter 5

---

# Question 237

**Type:** mc

What does predict() return?

**Options:**
- A) Probabilities
- B) Predicted class labels
- C) Confidence scores
- D) Feature importances

**Correct:** B

**Explanation:** Returns predicted class labels as array.

**Chapter:** Predictions

---

# Question 238

**Type:** mc

What does predict_proba() return?

**Options:**
- A) Class labels
- B) Probability estimates for each class
- C) Binary predictions
- D) Confidence intervals

**Correct:** B

**Explanation:** Returns probability estimates for each class as 2D array.

**Chapter:** Probability predictions

---

# Question 239

**Type:** mc

What does the solver parameter control?

**Options:**
- A) Data preprocessing
- B) Optimization algorithm to use
- C) Number of iterations
- D) Learning rate

**Correct:** B

**Explanation:** Specifies optimization algorithm like lbfgs, liblinear, newton-cg, or sag.

**Chapter:** Optimization algorithm

---

# Question 240

**Type:** mc

What is the default solver?

**Options:**
- A) 'newton-cg'
- B) 'lbfgs'
- C) 'sag'
- D) 'sgd'

**Correct:** B

**Explanation:** Default solver is lbfgs, effective for most problems.

**Chapter:** Default configuration

---

# Question 241

**Type:** mc

How do you import DecisionTreeClassifier?

**Options:**
- A) from sklearn import DecisionTreeClassifier
- B) from sklearn.tree import DecisionTreeClassifier
- C) from sklearn.models import DecisionTreeClassifier
- D) import sklearn.DecisionTreeClassifier

**Correct:** B

**Explanation:** Import DecisionTreeClassifier from sklearn.tree for decision tree models.

**Chapter:** Chapter 16

---

# Question 242

**Type:** mc

What is the criterion parameter?

**Options:**
- A) Stopping criterion
- B) Function to measure split quality (gini or entropy)
- C) Pruning method
- D) Tree depth

**Correct:** B

**Explanation:** Specifies function to measure split quality: gini or entropy.

**Chapter:** Split criterion

---

# Question 243

**Type:** mc

What are valid criterion values?

**Options:**
- A) 'mse', 'mae'
- B) 'gini', 'entropy'
- C) 'accuracy', 'precision'
- D) 'min', 'max'

**Correct:** B

**Explanation:** Valid values: 'gini' for Gini impurity, 'entropy' for information gain.

**Chapter:** Gini vs Entropy

---

# Question 244

**Type:** mc

What is max_depth?

**Options:**
- A) Maximum features
- B) Maximum tree depth
- C) Maximum samples
- D) Maximum nodes

**Correct:** B

**Explanation:** Limits maximum depth of tree to prevent overfitting.

**Chapter:** Chapter 17

---

# Question 245

**Type:** mc

What is min_samples_split?

**Options:**
- A) Minimum in leaf
- B) Minimum samples to split a node
- C) Minimum depth
- D) Minimum features

**Correct:** B

**Explanation:** Minimum samples required in node to allow splitting.

**Chapter:** Pre-pruning

---

# Question 246

**Type:** mc

What is min_samples_leaf?

**Options:**
- A) Minimum to split
- B) Minimum samples in a leaf node
- C) Minimum depth
- D) Minimum trees

**Correct:** B

**Explanation:** Minimum samples that must be present in leaf nodes.

**Chapter:** Pre-pruning

---

# Question 247

**Type:** mc

What is min_impurity_decrease?

**Options:**
- A) Maximum impurity
- B) Minimum decrease in impurity to split
- C) Impurity function
- D) Leaf impurity

**Correct:** B

**Explanation:** Split only if it decreases impurity by this minimum amount.

**Chapter:** Pre-pruning

---

# Question 248

**Type:** mc

What is ccp_alpha?

**Options:**
- A) Learning rate
- B) Complexity parameter for pruning
- C) Split criterion
- D) Feature importance threshold

**Correct:** B

**Explanation:** Complexity parameter for cost-complexity pruning; higher values mean more pruning.

**Chapter:** Post-pruning

---

# Question 249

**Type:** mc

How do you access feature_importances_?

**Options:**
- A) model.importances
- B) model.feature_importances_
- C) model.get_importances()
- D) model.importance

**Correct:** B

**Explanation:** Access feature_importances_ attribute after fitting tree.

**Chapter:** Feature importance

---

# Question 250

**Type:** mc

What does cost_complexity_pruning_path() return?

**Options:**
- A) Pruned tree
- B) Alpha values and corresponding impurities
- C) Best alpha
- D) Pruning decisions

**Correct:** B

**Explanation:** Returns alphas and corresponding impurities for choosing ccp_alpha.

**Chapter:** CCP implementation

---

