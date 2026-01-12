# from sklearn.model_selection import cross_val_score
# import numpy as np
# # Range of k values to try
# k_range = range(1, 21)
# cv_scores = []
# # Evaluate each k using 5-fold cross-validation
# for k in k_range:
# knn = KNeighborsClassifier(n_neighbors=k)
# scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
# cv_scores.append(scores.mean())
# # Plot accuracy vs. k
# plt.figure(figsize=(8, 5))
# plt.plot(k_range, cv_scores, marker='o')
# plt.title("k-NN Cross-Validation Accuracy vs k")
# plt.xlabel("Number of Neighbors: k")
# plt.ylabel("Cross-Validated Accuracy")
# plt.grid(True)
# plt.show()
# # Best k
# best_k = k_range[np.argmax(cv_scores)]
# print(f"Best k from cross-validation: {best_k}")

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Range of k values
k_range = range(1, 21)
cv_scores = []

# # 5-fold cross-validation for each k
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
#     cv_scores.append(scores.mean())

# # Plot accuracy vs k
# plt.figure(figsize=(8, 5))
# plt.plot(k_range, cv_scores, marker='o')
# plt.title("k-NN Cross-Validation Accuracy vs k")
# plt.xlabel("Number of Neighbors (k)")
# plt.ylabel("Cross-Validated Accuracy")
# plt.grid(True)
# plt.show()

# # Best k value
# best_k = k_range[np.argmax(cv_scores)]
# print("Best k from cross-validation:", best_k)      
  