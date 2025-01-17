# Thera Bank Personal Loan Campaign Classification

## Objective
The goal of this analysis is to **predict the likelihood** of a liability customer accepting a personal loan offer from Thera Bank. 
The **target variable** is:
- **Personal Loan**: Binary classification (1 = Loan accepted, 0 = Loan not accepted).

---

## Dataset Overview
The dataset contains various customer attributes that help predict personal loan acceptance. Below is a summary of the key features:

| **Feature**         | **Description**                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| **ID**              | Customer ID (unique identifier)                                                |
| **Age**             | Age of the customer (in years)                                                 |
| **Experience**      | Number of years of professional experience                                     |
| **Income**          | Annual income of the customer (in thousands)                                   |
| **ZIP Code**        | Home address ZIP code                                                          |
| **Family**          | Family size of the customer                                                    |
| **CCAvg**           | Average monthly credit card spending (in thousands)                           |
| **Education**       | Education level (1 = Undergrad, 2 = Graduate, 3 = Advanced/Professional)       |
| **Mortgage**        | Value of house mortgage (if any, in thousands)                                |
| **Personal Loan**   | Did the customer accept the personal loan? (1 = Yes, 0 = No)                   |
| **Securities Account** | Does the customer have a securities account? (1 = Yes, 0 = No)              |
| **CD Account**      | Does the customer have a certificate of deposit (CD) account? (1 = Yes, 0 = No)|
| **Online**          | Does the customer use internet banking? (1 = Yes, 0 = No)                     |
| **CreditCard**      | Does the customer have a credit card with the bank? (1 = Yes, 0 = No)          |

---

## Model and Analysis
**Logistic Regression** was used to predict whether a customer would accept a personal loan.

### Code Example:
```python
from sklearn.linear_model import LogisticRegression   # Importing logistic regression from scikit-learn

model = LogisticRegression(random_state=7)  # Assigning a variable for the algorithm

model.fit(xtrain, ytrain)  # Training the model

y_predict = model.predict(xtest)  # Model predicts y values based on the test dataset

print("Training accuracy:", model.score(xtrain, ytrain))  # Training accuracy
print("Testing accuracy:", model.score(xtest, ytest))  # Testing accuracy
```
## Training and Testing Accuracy:
- Training Accuracy: 0.9537
- Testing Accuracy: 0.9613

## Model Evaluation

### Confusion Matrix:
- Predicted: Yes	Predicted: No
- Actual: Yes	92 (True Positive)	46 (False Negative)
- Actual: No	12 (False Positive)	1350 (True Negative)
- Key Metrics
  
- Recall: 0.6667
Measures the proportion of actual positive cases (loan acceptance) that were correctly identified by the model.

- Precision: 0.8846
Measures the proportion of predicted positive cases that were actually positive.

- F1-Score: 0.7603
Harmonic mean of Precision and Recall, balancing both metrics.

- ROC-AUC: 0.8289
Measures the area under the receiver operating characteristic curve, indicating the model's ability to discriminate between classes.

## Conclusion

- Education, Experience, and CD Account are the most important features for predicting loan acceptance.
Age and Mortgage have minimal impact on loan acceptance predictions.

- Focus on Recall:
Since the objective of this campaign is to maximize the number of customers accepting the loan, Recall is a priority. This means identifying as many customers who will accept the loan as possible, even at the cost of more false positives.

## Libraries:
- scikit-learn (for machine learning)
- matplotlib, seaborn (for visualization)
- pandas, numpy (for data manipulation)
