import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# our goal is to maximize
best = max

### DATA LOADING ###

# Open the file
filename = 'polimi.csv'
df = pd.read_csv(filename)

# first column is the id (useless), then we have the metric and all the
# parameters
y = df.iloc[:, 1]
X = df.iloc[:, 2:]

# I like to normalize performance data using NPI, defined as achieved
# improvement over potential improvement:
# npi = (y - baseline) / (best - baseline)
# the baseline is the first line
y = (y - y[0]) / (best(y) - y[0])
# NPI should be maximized even if the original problem is minimize
# missing values (if any) represent failures, use a bad npi
y.iloc[y.isnull()] = -1
y = np.clip(y, -1, None)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# I want to create a random forest regressor on this data
# first we need to transform categorical features
# Other examples: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html
categorical_columns = X.dtypes == 'object'
numerical_columns = X.dtypes != 'object'
categorical_encoder = OneHotEncoder()
preprocessing = ColumnTransformer([('cat', categorical_encoder, categorical_columns)],
                                  remainder='passthrough')


### MODEL CREATION ###

# now we can fit the RF
rf = Pipeline([
    ('preprocess', preprocessing),
    ('regressor', RandomForestRegressor(random_state=42))
])
rf.fit(X_train, y_train)
print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))


### SOLUTION 1 ###

# The simplest solution is to use the RF feature importance
# this is some boilerplate code to print the feature names, nothing relevant
ohe = (rf.named_steps['preprocess']
         .named_transformers_['cat'])
feature_names = ohe.get_feature_names(input_features=X.columns[categorical_columns])
feature_names = np.r_[feature_names, X.columns[numerical_columns]]

tree_feature_importances = (
    rf.named_steps['regressor'].feature_importances_)
sorted_idx = tree_feature_importances.argsort()

y_ticks = np.arange(0, len(feature_names))
fig, ax = plt.subplots()
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances (MDI)")
fig.tight_layout()
plt.show()


### SOLUTION 2 ###

# A better way: permutation importance
#https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
result = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()


### SOLUTION 3 ###

# In this way, however, we might select parameters which are important to
# degrade performace!
# What we are really interested in are parameters which INCREASE the
# performance
# We can measure this by changing one parameter at a time, starting from the
# baseline configuration, and predict the score
best_idx = np.argmax(y)
best_x = X.iloc[best_idx, :]
bsl_x = X.iloc[0, :]
xs = []
for param in X.columns:
    x = bsl_x.copy()
    x[param] = best_x[param]
    xs.append(x)

# The baseline has an NPI of 0 by definition, so the prediction is the gain
gains = rf.predict(xs)
sorted_idx = gains.argsort()
y_ticks = np.arange(0, len(X.columns))
fig, ax = plt.subplots()
ax.barh(y_ticks, gains[sorted_idx])
ax.set_yticklabels(X.columns[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Gain from baseline to best")
fig.tight_layout()
plt.show()


### SOLUTION 4 ###
# The previous solution does not consider the *combined* effect of multiple
# parameters.
# To do that, we can change one parameter at a time, without restarting from
# the baseline.
# We can start from the importance we just computed, and see whether the
# combined effect gives a new ordering, and iterate.
# We have no assurance of convergence

new_ordering = True
sorted_cols = X.columns
iters = 0
while new_ordering and iters < 100:
    new_ordering = False
    iters += 1
    # we have a new param ordering, build the points
    xs = []
    x = bsl_x.copy()
    sorted_cols = sorted_cols[sorted_idx[::-1]]
    for param in sorted_cols:
        x = x.copy()
        x[param] = best_x[param]
        xs.append(x)

    # This time, to compute the gain we need to compute the difference for the
    # single parameter
    predictions = rf.predict(xs)
    gains = np.ediff1d(predictions, to_begin=predictions[0])
    old_idx = sorted_idx
    sorted_idx = gains.argsort()
    if any(old_idx != sorted_idx):
        new_ordering = True

# Use the final ordering, do not recompute the gains
xs = []
x = bsl_x.copy()
sorted_cols = sorted_cols[sorted_idx[::-1]]
for param in sorted_cols:
    x = x.copy()
    x[param] = best_x[param]
    xs.append(x)
predictions = rf.predict(xs)

y_ticks = np.arange(0, len(X.columns))
fig, ax = plt.subplots()
ax.barh(y_ticks, gains[sorted_idx], color='blue', height=0.4)
ax.barh(y_ticks+0.4, predictions[::-1], color='red', height=0.4)
ax.set_yticklabels(sorted_cols[::-1])
ax.set_yticks(y_ticks)
ax.set_title("Combined gain from baseline to best. Single parameter contribution in blue, overall in red")
fig.tight_layout()
plt.show()
