# Machine Learning Algorithms

7 ML algorithms implemented in Python using scikit-learn.

---

## 1. K-Means Clustering

**Theory:** Unsupervised learning algorithm that groups data into K clusters. Randomly initializes K centroids, assigns each point to nearest centroid, then updates centroids by averaging all points in each cluster. Repeats until convergence.

**Process:**
1. Choose K (number of clusters)
2. Initialize K random centroids
3. Assign each point to nearest centroid
4. Update centroids to mean of cluster points
5. Repeat until no change

**Use Case:** Customer segmentation, image compression, document clustering.

---

## 2. Logistic Regression

**Theory:** Binary classification algorithm. Uses sigmoid function to map linear combination of features to probability between 0 and 1. Decision boundary is a straight line (hyperplane).

**Equation:** P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + ...))

**Cost Function:** Cross-entropy loss (log loss)

**Use Case:** Medical diagnosis, spam detection, credit scoring.

---

## 3. Support Vector Machine (SVM)

**Theory:** Finds optimal hyperplane that maximizes margin between classes. Support vectors are the critical points closest to the hyperplane. Can use kernels to handle non-linear data.

**Key Concepts:**
- **Hyperplane:** Decision boundary separating classes
- **Margin:** Distance between hyperplane and support vectors
- **Kernel:** Functions that transform data to higher dimension (linear, RBF, polynomial)

**Use Case:** Text classification, image recognition, face detection.

---

## 4. Decision Tree & Random Forest

**Theory:**

**Decision Tree:** Tree-like model making decisions by splitting data based on feature values. Splits based on information gain (entropy) or Gini impurity. Prone to overfitting.

**Random Forest:** Ensemble of multiple decision trees. Each tree trained on random subset of data (bootstrap) and considers random features at each split. Reduces overfitting and improves accuracy.

**Key Differences:**
- Decision Tree: Single tree, fast, prone to overfitting
- Random Forest: Multiple trees, more robust, better generalization

**Use Case:** Fraud detection, medical diagnosis, customer churn prediction.

---

## 5. DBSCAN Clustering

**Theory:** Density-based clustering algorithm. Forms clusters based on density of points. Does not require specifying number of clusters beforehand. Can identify outliers as noise.

**Parameters:**
- **eps (epsilon):** Maximum distance between two points to be neighbors
- **min_samples:** Minimum points needed to form a dense region

**Concepts:**
- **Core point:** Has at least min_samples neighbors within eps
- **Border point:** Within eps of a core point but not a core point itself
- **Noise point:** Neither core nor border point

**vs K-means:**
- K-means: Must specify K, assumes spherical clusters
- DBSCAN: No need to specify K, can find arbitrary shapes

**Use Case:** Geographic clustering, anomaly detection, market basket analysis.

---

## 6. Linear Regression

**Theory:** Supervised learning algorithm for predicting continuous target variable. Finds best-fit line (hyperplane) that minimizes sum of squared errors between predicted and actual values.

**Equation:** y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

**Metrics:**
- **MSE:** Mean Squared Error (penalizes large errors)
- **RMSE:** Square root of MSE (same unit as target)
- **MAE:** Mean Absolute Error (robust to outliers)
- **R²:** Coefficient of determination (0 to 1, higher is better)

**Assumptions:** Linear relationship, no multicollinearity, homoscedasticity, normality of residuals.

**Use Case:** House price prediction, sales forecasting, risk assessment.

---

## 7. Neural Network (MLP)

**Theory:** Inspired by biological neurons. Feed-forward network with input, hidden, and output layers. Each neuron applies weighted sum followed by activation function.

**Architecture:**
- **Input Layer:** Features
- **Hidden Layers:** Process information through weights
- **Output Layer:** Predictions

**Key Concepts:**
- **Neuron:** Weighted sum + activation
- **Activation Function:** ReLU, Sigmoid, Tanh (adds non-linearity)
- **Backpropagation:** Algorithm to update weights by minimizing loss
- **Gradient Descent:** Optimization algorithm

**Training Process:**
1. Forward pass: Input → hidden → output
2. Calculate loss
3. Backward pass: Update weights
4. Repeat until convergence

**Use Case:** Image recognition, NLP, speech recognition, game AI.

---

## Setup

```bash
cd /Users/sapatmohit18/Desktop/ML
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn
```

## Run All

```bash
.venv/bin/python 1_kmeans_clustering.py
.venv/bin/python 2_logistic_regression.py
.venv/bin/python 3_svm_classification.py
.venv/bin/python 4_decision_tree_random_forest.py
.venv/bin/python 5_dbscan_clustering.py
.venv/bin/python 6_linear_regression.py
.venv/bin/python 7_neural_network.py
```

## Files

| # | File | Algorithm | Task |
|---|------|-----------|------|
| 1 | `1_kmeans_clustering.py` | K-Means | Clustering |
| 2 | `2_logistic_regression.py` | Logistic Regression | Binary Classification |
| 3 | `3_svm_classification.py` | SVM | Classification |
| 4 | `4_decision_tree_random_forest.py` | Decision Tree + Random Forest | Classification |
| 5 | `5_dbscan_clustering.py` | DBSCAN | Density-based Clustering |
| 6 | `6_linear_regression.py` | Linear Regression | Regression |
| 7 | `7_neural_network.py` | MLP Neural Network | Classification |

## Results

- K-means: 3 clusters
- Logistic Regression: 100% accuracy
- SVM: 80% accuracy
- Decision Tree: 100% accuracy
- Random Forest: 100% accuracy
- Linear Regression: RMSE 17.4
- Neural Network: 100% accuracy