# Thera Bank Personal Loan Campaign Classification

The objective of this analysis is to predict the likelihood of a liability customer buying a personal loan offered by Thera Bank. The goal is to understand which features of a customer influence their decision to accept a personal loan, and to improve marketing strategies for future campaigns.

The target variable in this dataset is:
Personal Loan (binary classification: 1 for accepting the loan, 0 for not accepting the loan).

# Dataset Overview:
The dataset contains various customer attributes to predict whether or not they would accept a personal loan offer. Below is a brief explanation of the attributes used in the model:

ID: Customer ID (unique identifier)

Age: Age of the customer in years

Experience: Number of years of professional experience

Income: Annual income of the customer (in thousands)

ZIP Code: Home address ZIP code

Family: Family size of the customer

CCAvg: Average spending on credit cards per month (in thousands)

Education: Education level (1 = Undergrad, 2 = Graduate, 3 = Advanced/Professional)

Mortgage: Value of house mortgage (if any, in thousands)

Personal Loan: Did the customer accept the personal loan (1 = Yes, 0 = No)

Securities Account: Does the customer have a securities account with the bank (1 = Yes, 0 = No)

CD Account: Does the customer have a certificate of deposit (CD) account with the bank (1 = Yes, 0 = No)

Online: Does the customer use internet banking (1 = Yes, 0 = No)

CreditCard: Does the customer have a credit card with the bank (1 = Yes, 0 = No)

Model and Analysis:
Logistic Regression is used to predict whether a customer will accept a personal loan.

```r
from sklearn.linear_model import LogisticRegression   # Importing logistic regression from scikit-learn

model = LogisticRegression(random_state=7)  # Assigning a variable for the algorithm

model.fit(xtrain, ytrain)  # Training the model

y_predict = model.predict(xtest)  # Model predicts y values based on the test dataset

print("Training accuracy:", model.score(xtrain, ytrain))  # Training accuracy
print("Testing accuracy:", model.score(xtest, ytest))  # Testing accuracy
```
Training and Testing Accuracy:
Training Accuracy: 0.9537
Testing Accuracy: 0.9613

# Model Evaluation:

1. Confusion Matrix:
A confusion matrix provides a summary of the model's predictions compared to the actual values.

True Positive (TP): 92 — Predicted the customer would take a loan, and they did.
False Positive (FP): 12 — Predicted the customer would take a loan, but they did not.
True Negative (TN): 1350 — Predicted the customer would not take a loan, and they did not.
False Negative (FN): 46 — Predicted the customer would not take a loan, but they did.


2. Key Metrics:
The following performance metrics were calculated:

### Recall: 0.6667

Measures the proportion of actual positive cases (loan acceptance) that were correctly identified by the model.

### Precision: 0.8846

Measures the proportion of predicted positive cases that were actually positive.

### F1-Score: 0.7603

Harmonic mean of Precision and Recall, balancing both metrics.

### ROC-AUC: 0.8289

Measures the area under the receiver operating characteristic curve, indicating the model's ability to discriminate between the classes.

# Conclusion:
Key Insights:
Education, Experience, and CD Account are the most important features contributing to predicting whether a customer will accept a personal loan.

Age and Mortgage seem to have minimal impact on the prediction of loan acceptance.

Focus on Recall: Since the objective of this campaign is to maximize the number of customers accepting the loan, the bank is most interested in recall. This means the model should identify as many customers who will accept the loan as possible, even if it leads to a higher number of false positives.

# Requirements:
Python 3.x

Libraries:

scikit-learn (for machine learning)

matplotlib, seaborn (for visualization)

pandas, numpy (for data manipulation)
