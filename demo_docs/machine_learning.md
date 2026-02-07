# Machine Learning Concepts

## Supervised vs Unsupervised Learning

Machine learning algorithms fall into two primary categories. Supervised learning trains on labeled data where each input has a known output. The model learns to map inputs to outputs and generalizes to unseen examples. Unsupervised learning works with unlabeled data, discovering hidden patterns and structures without predefined targets.

Semi-supervised learning combines both approaches, using a small amount of labeled data alongside a larger pool of unlabeled data. Reinforcement learning is a third paradigm where an agent learns optimal actions through trial and error, receiving rewards or penalties from its environment.

## Regression and Classification

Regression predicts continuous numerical values. Linear regression fits a straight line to the data by minimizing the sum of squared errors. Polynomial regression captures nonlinear relationships by adding higher-degree terms. Ridge and Lasso regression add regularization penalties to prevent overfitting and handle multicollinearity.

Classification predicts discrete categories. Logistic regression models the probability of binary outcomes using a sigmoid function. Decision trees split data based on feature thresholds, creating interpretable rule-based models. Random forests combine multiple decision trees through bagging to reduce variance. Support vector machines find the optimal hyperplane that maximizes the margin between classes.

## Neural Networks

Neural networks consist of layers of interconnected nodes. Each node applies a weighted sum followed by a nonlinear activation function. The input layer receives raw features, hidden layers extract increasingly abstract representations, and the output layer produces predictions.

Backpropagation calculates gradients of the loss function with respect to each weight, and gradient descent updates weights to minimize the loss. Common activation functions include ReLU, sigmoid, and tanh. Deep learning refers to networks with many hidden layers, capable of learning complex hierarchical representations.

## Overfitting and Regularization

Overfitting occurs when a model memorizes training data rather than learning generalizable patterns. Signs include high training accuracy paired with poor validation performance. Underfitting is the opposite problem, where the model is too simple to capture the underlying relationships.

Regularization techniques combat overfitting. L1 regularization encourages sparsity by penalizing the absolute values of weights. L2 regularization penalizes large weight values, producing smoother models. Dropout randomly deactivates neurons during training, forcing the network to learn redundant representations. Early stopping halts training when validation performance begins to degrade.

## Cross-Validation and Evaluation

Cross-validation provides a robust estimate of model performance. K-fold cross-validation splits the data into k subsets, training on k-1 folds and validating on the remaining fold, rotating through all combinations. Stratified k-fold preserves class proportions in each fold.

Evaluation metrics depend on the task. Accuracy measures overall correctness but is misleading for imbalanced datasets. Precision and recall capture the tradeoff between false positives and false negatives. The F1 score is the harmonic mean of precision and recall. ROC-AUC measures the model's ability to distinguish between classes across all threshold values.

## Feature Engineering

Feature engineering transforms raw data into informative inputs for machine learning models. Numerical features may benefit from scaling (standardization or min-max normalization), log transforms for skewed distributions, or binning into discrete categories.

Categorical features are encoded using one-hot encoding, label encoding, or target encoding. Text features are converted to numerical representations through bag-of-words, TF-IDF, or word embeddings. Feature selection identifies the most relevant features using correlation analysis, mutual information, or model-based importance scores.
