import numpy as np


lam_arrays = np.linspace(10 ** (-6), 20, 200)  # Creat array use for saving different 20000 lambda value (range from 10^-6 to 30) by unifrom distribution
number = lam_arrays.size
def lasso_solve(A, d, la_array):
    # ista_solve_hot: Iterative soft-thresholding for multiple values of
    # lambda with hot start for each case - the converged value for the previous
    # value of lambda is used as an initial condition for the current lambda.
    # this function solves the minimization problem
    # Minimize |Ax-d|_2^2 + lambda*|x|_1 (Lasso regression)
    # using iterative soft-thresholding.
    max_iter = 10 ** 4
    tol = 10 ** (-3)
    tau = 1 / np.linalg.norm(A) ** 2
    n = A.shape[1]
    w = np.zeros((n, 1))
    num_lam = len(la_array)
    X = np.zeros((n, num_lam))
    for i, each_lambda in enumerate(la_array):
        for j in range(max_iter):
            z = w - tau * (A.T @ (A @ w - d))
            w_old = w
            w = np.sign(z) * np.clip(np.abs(z) - tau * each_lambda / 2, 0, np.inf)
            X[:, i:i + 1] = w
            if np.linalg.norm(w - w_old) < tol:
                break
    return X



# use the 10 cross vaildation to split the data
setindices = np.asarray([[0, 812], [812, 1624], [1624, 2436], [2436, 3248], [3248, 4060], [4060, 4872],[4872,5684],[5684,6497],[6497,7310],[7310,8123]])
holdoutindices = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],[6, 7],[7, 8],[8, 9],[9, 0]])
cases = len(holdoutindices)
index = np.asarray(range(8123))
errv2_l1 = np.zeros(cases)

# Find the best W and lumda to use by LASSO(L1)
for j in range(cases):
    # row indices of first validation set
    v1_ind = index[setindices[holdoutindices[j, 0], 0]:
                   setindices[holdoutindices[j, 0], 1]]
    # row indices of second validation set
    v2_ind = index[setindices[holdoutindices[j, 1], 0]:
                   setindices[holdoutindices[j, 1], 1]]

    # row indices of training set
    trn_ind = np.setdiff1d(np.setdiff1d(index, v1_ind), v2_ind);

    # validation set 1
    Av1 = features[v1_ind, :]
    bv1 = labels[v1_ind, :]

    # validation set 2
    Av2 = features[v2_ind, :]
    bv2 = labels[v2_ind, :]

    # training set
    At = features[trn_ind, :]
    bt = labels[trn_ind, :]

    # Use training data to learn classifier
    W = lasso_solve(At, bt, lam_arrays)
    Bhatv1 = Av1 @ W

    # Find best lambda value using first validation set, then evaluate
    errv1 = np.zeros(number)
    for i in range(number):
        target = bv1
        predict = Bhatv1[:, i]
        errv1[i] = np.mean(target!=predict)/target.size

    min_ind = np.argmin(errv1)
    errv2_l1[j] = np.mean(bv2!=Av2@W[:, min_ind])/bv2.size
print("The average error rate by LASSO is " + str(np.average(errv2_l1)))