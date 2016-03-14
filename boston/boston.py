import timeit

import pandas
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

boston = load_boston()
boston_df = pandas.DataFrame(boston.data, columns=boston.feature_names)
boston_df['Target'] = boston.target

# print boston_df.describe()
pearson = boston_df.corr(method='pearson')
print pearson['Target'].abs().sort_values(inplace=False)

lr = linear_model.LinearRegression()
X = boston.data
y = boston.target

predictors = ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX', 'NOX']

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(lr, boston_df[predictors], y, cv=10)
scores = cross_val_score(lr, boston.data, boston_df["Target"], cv=3)



# fig, ax = plt.subplots()
# ax.scatter(y, predicted)
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()

lr.fit(boston_df[predictors], boston_df['Target'])
print lr.score(boston_df[predictors], y, sample_weight=None)