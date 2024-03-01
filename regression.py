"""
Implement linear regression example using built-in scikit-learn datasets
"""
from sklearn import datasets
import pandas as pd
import numpy as np
import statsmodels.api as stmod
import sklearn.linear_model as lmodel
import sklearn.metrics as smet

if __name__ == '__main__':

	"""
	LINEAR REGRESSION
	"""
	"""
	# get data into the "natural" input for practice having to parse things
	data = datasets.load_diabetes()
	
	# put data in pandas dataframe
	target = data['target'].reshape((-1, 1))
	joined = np.concatenate((data['data'], target), axis=1)
	features = data['feature_names']
	features.append('target')

	df = pd.DataFrame(data=joined, columns=features)

	X = df.loc[:, df.columns != 'target']
	Y = df['target']

	# add column of ones so it calculates the intercept
	X = stmod.add_constant(X)

	model = stmod.OLS(Y, X).fit()

	predict_Y = model.predict(X)
	mse = stmod.tools.eval_measures.mse(predict_Y, Y)
	print('MSE:', mse) # = 3859.69
	# analyze p-values
	print(model.summary()) # R-squared: 0.518

	sign = ['sex', 'bmi', 'bp', 's5']
	X = df.loc[:,sign]
	Y = df.loc[:,'target']
	X = stmod.add_constant(X)

	model = stmod.OLS(Y, X).fit()
	predict_Y = model.predict(X)
	mse = stmod.tools.eval_measures.mse(predict_Y, Y)
	print('MSE:', mse) # 3043.38 (slightly better)
	print(model.summary()) # R-squared: 0.487

	# better model with sklearn instead
	model = lmodel.LogisticRegression().fit(X, Y)
	predict_Y = model.predict(X)
	mse = smet.mean_squared_error(predict_Y, Y)
	print('MSE:', mse)
	"""
	
	"""
	LOGISTIC REGRESSION
	"""
	
	data = datasets.load_breast_cancer()
	# put into pandas dataframe for practice
	target = data['target'].reshape((-1, 1))
	joined = np.concatenate((data['data'], target), axis=1)
	features = data['feature_names']
	features = np.append(features, 'target')

	df = pd.DataFrame(data=joined, columns=features)
	df['target'].astype('category')

	# making one model is timing out, need to make separate ones to check pvalues
	# good to do they by type, all float64, so just group in order
	#print(df.dtypes)
	g1 = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension']
	g2 = ['radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error']
	g3 = ['worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']

	X1 = df.loc[:, g1]
	X2 = df.loc[:, g2]
	X3 = df.loc[:, g3]
	Y = df.loc[:, 'target']

	model1 = stmod.Logit(Y, X1).fit()
	model2 = stmod.Logit(Y, X2).fit()
	model3 = stmod.Logit(Y, X3).fit()

	print(model1.summary())
	print(model2.summary())
	print(model3.summary())

	sign = ['mean texture', 'mean area', 'mean smoothness', 'mean concave points', 'radius error', 'texture error', 'area error', 'smoothness error', 'compactness error',
	'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst area', 'worst smoothness', 'worst concave points']

	X_final = df.loc[:, sign]
	model = stmod.Logit(Y, X_final).fit()
	predict_Y = model.predict(X_final)
	mse = stmod.tools.eval_measures.mse(predict_Y, Y)
	print('MSE:', mse) # 0.015
	print(model.summary())

	sign = ['worst radius', 'worst texture', 'worst area']

	X = df.loc[:, sign]
	model = stmod.Logit(Y, X).fit()
	predict_Y = model.predict(X)
	mse = stmod.tools.eval_measures.mse(predict_Y, Y)
	print('MSE:', mse) # 0.0495
	print(model.summary())
	


"""
Notes

--p-value for a variable less than significance level, data rejects null hypothesis
----null hypothesis: no correlation between independent ad dependent variable
--p-values can help determine which variables to include in final model

--sign of regression coefficient says positive or negative correlation
--unstandardized effect size
"""

	

	