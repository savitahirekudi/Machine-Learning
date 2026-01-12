
# import pandas as pd

# df = pd.read_csv(r"D:\sana\heart (1).csv")
# print(df.head())


# # Features and target
# X = df.drop('target', axis=1)
# y = df['target']

# # Display output
# print(df.head())
# print("\nX shape:", X.shape)
# print("y shape:", y.shape)


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# import numpy as np
# from sklearn.datasets import make_classification

# x,y=make_classification(
#     n_samples=1000,n_features=20,n_informative=10,n_classes=2,random_state=42)
# c_space=np.logspace(-5,8,15)
# param_grid={'C':c_space}

# logreg=LogisticRegression()

# logreg_cv=GridSearchCV(logreg,param_grid,cv=7)

# logreg_cv.fit(x,y)

# print("Tuned logistic Regression parameters:{}".format(logreg_cv.best_params_))
# print("Best score is{}".format(logreg_cv.best_score_))

# import numpy as np
# from sklearn.datasets import make_classification

# x,y=make_classification(n_samples=1000,n_features=20,n_informative=10,n_classes=2,random_state=32)

# from scipy.stats import randint
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import RandomizedSearchCV

# param_dist={
#     "max_depth":[3,None],
#     "max_features":randint(1,2),
#     "min_samples_leaf":randint(1,2),
#     "criterion":["gini","entropy"]
# }

# tree=DecisionTreeClassifier()
# tree_cv=RandomizedSearchCV(tree,param_dist,cv=5)
# tree_cv.fit(x,y)

# print("Tuned logistic Regression parameters:{}".format(tree_cv.best_params_))
# print("Best score is{}".format(tree_cv.best_score_))

import mathplotlib.pyplot as plt

plt.figure()
plt.plot(fpr,tpr,color  curve ')













































