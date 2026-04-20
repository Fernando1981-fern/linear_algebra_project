# Linear Algebra Project: Sure Tomorrow Insurance Company Analysis

## Overview

This project demonstrates the practical application of linear algebra and machine learning techniques to solve real-world business problems for the Sure Tomorrow insurance company. The notebook contains four interconnected tasks that showcase fundamental linear algebra concepts and their implementation in Python.

---

## Project Tasks

### Task 1: Similar Customers (k-Nearest Neighbors)

**Objective:** Find customers similar to a given customer to support targeted marketing strategies.

**Linear Algebra Concepts:**
- **Distance Metrics:** Calculate distances between vector representations of customers
  - **Euclidean Distance:** The most common distance metric, calculated as the square root of the sum of squared differences between coordinates
  - **Manhattan Distance:** An alternative metric that sums the absolute differences between coordinates
  
- **Feature Scaling:** Data normalization using MaxAbsScaler to ensure fair distance calculations
  - Without scaling, features with larger ranges (e.g., income) disproportionately influence distance calculations
  - With scaling, all features contribute equally to the distance metric

**Implementation:**
- k-Nearest Neighbors (kNN) algorithm implemented using scikit-learn's `NearestNeighbors` class
- Tested with four combinations:
  1. Original data + Euclidean distance
  2. Original data + Manhattan distance
  3. Scaled data + Euclidean distance
  4. Scaled data + Manhattan distance
  
**Key Findings:**
- Data scaling significantly affects kNN results by normalizing feature contributions
- Manhattan and Euclidean distances produce different neighbor selections
- Proper feature scaling is essential for accurate distance-based algorithms

---

### Task 2: Insurance Benefit Classification (Binary Classification)

**Objective:** Predict whether a customer will receive any insurance benefit using classification models.

**Linear Algebra & ML Concepts:**
- **Binary Classification:** Target variable = 1 if customer receives any benefit, 0 otherwise
- **k-Nearest Neighbors Classifier:** Classification using the k-nearest neighbors
- **Probability Calculation:** 
  $$P\{\text{insurance benefit received}\} = \frac{\text{number of clients received any insurance benefit}}{\text{total number of clients}}$$
  
- **Dummy Models:** Baseline models for comparison
  - Always predict 0 (probability = 0)
  - Predict 1 with probability equal to base rate
  - Always predict 1 with probability 0.5
  - Always predict 1 (probability = 1)

- **F1 Score Metric:** Harmonic mean of precision and recall, useful for imbalanced datasets
  $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Implementation:**
- Split data into 70% training and 30% testing sets
- Train KNN classifier with k ranging from 1 to 10
- Evaluate both original and scaled data
- Compare with dummy model baseline

**Key Findings:**
- KNN classifier substantially outperforms dummy models when using properly scaled data
- Feature scaling improves F1 scores significantly
- The optimal k value balances bias and variance

---

### Task 3: Linear Regression for Benefit Count Prediction

**Objective:** Predict the number of insurance benefits a customer will receive.

**Linear Algebra Concepts:**

The linear regression task is formulated in matrix form as:
$$y = Xw$$

Where:
- **X** = feature matrix (n × m), each row is a customer, each column is a feature
- **y** = target vector (n × 1), the number of insurance benefits
- **ŷ** = predicted target vector
- **w** = weight vector (m × 1), the model parameters

**Optimization Objective:**

Minimize the Mean Squared Error (MSE) or L2 distance:
$$\min_w \text{MSE}(Xw, y) = \min_w \|Xw - y\|^2$$

**Analytical Solution (Normal Equation):**
$$w = (X^T X)^{-1} X^T y$$

This closed-form solution finds the weights that minimize the MSE by:
1. Computing the design matrix X augmented with a bias term (column of ones)
2. Calculating the matrix product $X^T X$
3. Computing the matrix inverse $(X^T X)^{-1}$
4. Performing matrix multiplications to obtain w

**Predictions:**
$$\hat{y} = X_{val} w$$

**Evaluation Metrics:**
- **RMSE (Root Mean Squared Error):** 
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
  
- **R² Score:** Coefficient of determination, measuring the proportion of variance explained by the model
  $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

**Implementation:**
- Custom `MyLinearRegression` class implementing the normal equation
- Tested on both original and scaled data
- Train/test split: 70% training, 30% validation

**Key Findings:**
- RMSE and R² are identical for original and scaled data
- Linear regression is scale-invariant for these metrics

---

### Task 4: Data Obfuscation (Privacy Protection)

**Objective:** Protect customer personal data through transformation without compromising model performance.

**Linear Algebra Concept: Matrix Multiplication for Data Transformation**

The obfuscation method multiplies the feature matrix by an invertible transformation matrix:
$$X' = X \times P$$

Where:
- **X** = original feature matrix
- **P** = invertible transformation matrix (4 × 4)
- **X'** = obfuscated (transformed) feature matrix

**Invertibility Check:**
A matrix P is invertible if and only if its determinant is non-zero:
$$\det(P) \neq 0 \implies P \text{ is invertible}$$

**Data Recovery:**
The original data can be recovered using the inverse matrix:
$$X = X' \times P^{-1}$$

**Analytical Proof: Obfuscation Preserves Linear Regression**

When linear regression is trained on obfuscated data $X' = X P$, the new weights $w_P$ are:

$$w_P = [(XP)^T(XP)]^{-1}(XP)^T y$$

**Step-by-step derivation using matrix properties:**

1. **Expand the matrix product:**
   $$w_P = [P^T X^T X P]^{-1} P^T X^T y$$

2. **Apply matrix inversion property $(ABC)^{-1} = C^{-1}B^{-1}A^{-1}$:**
   $$w_P = P^{-1}(X^T X)^{-1}(P^T)^{-1} P^T X^T y$$

3. **Apply transpose inverse property $(P^T)^{-1} = (P^{-1})^T$:**
   $$w_P = P^{-1}(X^T X)^{-1} X^T y$$

4. **Recognize original weights w:**
   $$w_P = P^{-1} w$$

**Prediction Preservation:**
$$\hat{y}' = X' w_P = (XP)(P^{-1}w) = X(PP^{-1})w = X I w = Xw = \hat{y}$$

The predictions remain identical because $PP^{-1} = I$ (identity matrix).

**Metric Invariance:**
$$\text{RMSE}(y, \hat{y}') = \text{RMSE}(y, \hat{y})$$

Since predictions are identical, all performance metrics (RMSE, R², etc.) remain unchanged.

**Implementation:**
- Create random 4 × 4 invertible matrix P
- Transform feature matrix: $X' = X \times P$
- Train linear regression models on both original and obfuscated data
- Verify that predictions and metrics are preserved

**Key Findings:**
- Data obfuscation successfully protects personal information
- Obfuscated data remains cryptographically secure (difficult to reverse without P)
- Linear regression performance is completely preserved
- RMSE and R² scores are identical for original and obfuscated datasets

---

## Matrix Properties Used

| Property | Formula |
|----------|---------|
| Distributivity | $A(B+C) = AB + AC$ |
| Non-commutativity | $AB \neq BA$ |
| Associative multiplication | $(AB)C = A(BC)$ |
| Multiplicative identity | $IA = AI = A$ |
| Inverse multiplication | $A^{-1}A = AA^{-1} = I$ |
| Inverse of product | $(AB)^{-1} = B^{-1}A^{-1}$ |
| Transpose of product | $(AB)^T = B^T A^T$ |

---

## Technologies & Libraries

- **NumPy:** Matrix operations, linear algebra computations
- **Pandas:** Data manipulation and preprocessing
- **Scikit-learn:** Machine learning algorithms and metrics
- **Seaborn:** Data visualization
- **Jupyter Notebook:** Interactive computational environment

---

## Conclusions

This project demonstrates the powerful synergy between linear algebra and machine learning:

1. **Distance Metrics & Feature Scaling:** Proper normalization is crucial for distance-based algorithms like kNN
2. **Classification:** KNN classification can significantly outperform baseline dummy models when data is properly scaled
3. **Linear Regression:** The matrix-based formulation of linear regression enables efficient, closed-form solutions
4. **Data Privacy:** Matrix transformations allow for effective data obfuscation without sacrificing model performance

The Sure Tomorrow insurance company can leverage these techniques to:
- Identify customer segments for targeted marketing
- Predict insurance benefit claims with high accuracy
- Estimate benefit counts reliably
- Protect customer privacy while maintaining model quality
