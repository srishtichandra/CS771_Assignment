import numpy as np
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses

	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0

    X_train = my_map(X_train)
    y_train = 2 * y_train - 1

    model = LogisticRegression(C=100, solver='lbfgs', max_iter=1000, tol=1e-3, penalty='l2')
    model.fit(X_train, y_train)

    coef = model.coef_.flatten()
    intercept = model.intercept_.flatten()

    w, b = np.array(coef), np.array(intercept)

    return w, b

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points

    feat = np.empty((X.shape[0], X.shape[1] * (X.shape[1] + 1) // 2), dtype=int)

    X = 1 - 2 * X

    cumulative_products = np.flip(np.cumprod(np.flip(X[:, :, None], axis=1), axis=1), axis=1)
    cumulative_products = cumulative_products.reshape(len(X), -1)
    
    m = cumulative_products.shape[1]

    x_ij = cumulative_products[:, :, None] * cumulative_products[:, None, :]

    mask = np.triu(np.ones((m, m), dtype=bool), k=1)
    
    x_ij_2d = x_ij[:, mask]
    x_ij_2d = x_ij_2d.reshape(-1, m * (m - 1) // 2)

    feat = np.concatenate((x_ij_2d, cumulative_products[:,:]), axis=1)

    return feat