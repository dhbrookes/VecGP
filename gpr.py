import numpy as np
import theano.tensor as T
import theano
from theano import config

__author__ = 'David Brookes'
__date__ = '8/14/17'

"""
Module for implementation of Gaussian Process Regression
"""


class GPR(object):
    """ Class for Gaussian Process Regression with any kernel

    This class implements a general Gaussian Process Regression (GPR) model
    for machine learning. It is build on the Theano framework for tensor
    operations. This class is designed to be as general as possible and
    makes very few assumptions about the input and output of the GP being
    modeled. The specificity comes entirely from the member kernel, which
    must be compatible with the database data.

    GP regression is a supervised, non-parametric learning technique that
    assumes the function, f, mapping inputs, X = {x_i}, to outputs, Y = {y_i},
    is drawn from a Gaussian Process, and thus any subset of function evaluations is
    normally distributed. The mean of this distribution is taken to be zero
    (in this implementation), and the covariance is given by the input kernel
    function, k(x, x'). The database of training points is then distributed
    according to
                            {f(x_i)} ~ N(0, K + nu*I)
    where {f(x_i)} is the vector of function evaluations (f(x_i) = y_i + nu, where
    nu is some amount of constant noise assumed in the data) and K
    is a matrix of pairwise kernel function evaluations, ie. K_ij = k(x_i, x_j).
    Then the expected value of the function at a set of novel points, {xstar_i}, given the
    database is given by the expression
                        E[ {f(xstar)} | X, Y] = Kstar * K^(-1) * Y
    where Kstar is the matrix of kernel evaluations of novel points to database points,
    i.e. Kstar_ij = k(xstar_i, x_i). This provides a means to perform predict ouputs
    at any point in input space. Note that the variance of the distribution at xstar
    can also be predicted easily. For more information on GPR models, see [1].

    This class has two main uses. The first is to make predictions using the scheme
    described above, giving the kernel, database of training points, and set of
    novel points for prediction. The major step here is calculating and inverting
    the kernel matrix. In certain cases, the theano implementation of kernels can
    cause memory issues if there are too many points in the database, so we
    have implemented a number of methods to construct the kernel matrix in batches.
    These methods break down the steps of prediction into many theano
    functions whose results can be concateneted over batches. Note that the
    concatenation occurs outside of this class, in independent functions.
    The second major use of the class is to optimize hyperparameters in the kernels.
    This is done by calculating the log likelihood of the database distribution and
    optimizing w.r.t the kernel parameters.  This should be done in batches as well,
    as thus uses stochastic gradient descent to perform the optimization.

    [1] C.E. Rasmussen and C.K.I Williams. MIT Press (2006)

    Args:
        kernel (BaseKernel): a kernel function
        nu (float): constant noise assumed in the data
        pred_std (bool): True if one wants to predict the variance as well as mean
        learn_rate (float): rate parameter for stochastic gradient descent
        opt_nu (bool): True if one wants to opimtize nu as well as kernel params
        build_batch_funcs (bool): if True, the necessary theano functions for building
                                  the kernel matrix will be built.

    Attributes:
        kernel_ (BaseKernel): kernel function
        learnRate_ (theano.shared float): rate parameter for stochastic gradient descent
        xrank_ (int): rank of tensor representing input points (ensures compatibility with data)
        nu_ (theano.shared float): constant noise assumed in data
        params_ (list of theano.shared floats): optimizable parameters
        batch_ (bool): build batch functions if True
        Kinvfunc_ (theano.function): function that constructs inverse kernel matrix given input points X
        Ksubfunc_ (theano.function): function that constructs kernel matrix given input points X (for batches)
        predFunc_ (theano.function): prediction
        addnufunc_ (theano.function): function that adds constant noise to diagonal of kernel matrix (for batches)
        invFunc_ (theano.function): function that inverts kernel matrix (for batches)
        batchPred_ (theano.function): function for performing predictions by constructing the kernel vector in batches
        optStep_ (theano.function): function that performs an optimization step

    """

    def __init__(self, kernel, nu=5e-4, pred_std=False, learn_rate=1e-8,
                 opt_nu=False, build_batch_funcs=False):
        self.kernel_ = kernel
        self.learnRate_ = theano.shared(learn_rate)
        self.xrank_ = self.kernel_.xrank_
        self.nu_ = theano.shared(nu, name='nu')
        self.params_ = kernel.get_params()
        self.batch_ = build_batch_funcs
        if opt_nu:
            self.params_ += [self.nu_]

        X = self._get_x_var('X')  # symbolic database inputs
        Y = T.dcol('Y')  # symbolic database outputs
        K = T.dmatrix('K')
        Kinv = T.dmatrix('Kinv')

        self.Kinvfunc_ = self._build_Kinv_func(X)
        self.predFunc_ = self._build_pred_func(X, Y, Kinv, pred_std=pred_std)

        if self.batch_:
            X1 = self._get_x_var('X1')
            X2 = self._get_x_var('X2')
            kstar = T.dmatrix('kstar')
            self.Ksubfunc_ = self._build_Ksub_func(X1, X2)
            self.addnufunc_ = self._build_add_nu_func(K)
            self.invFunc_ = self._build_inv_func(K)
            self.batchPred_ = self._build_batch_pred_func(kstar, Y, Kinv)

        self.optStep_ = self._build_opt_step_func()

    def _build_Kinv_func(self, X):
        """ Build theano function that constructs inverse kernel from data

        This method takes a symbolic variable X representing the tensor
        of database input points, and returns a theano function that calculates
        the inverse kernel matrix given a numpy array of actual (i.e. non-symbolic)
        input data.

        Args:
            X (theano TensorType): symbolic tensor of database inputs

        Returns:
            theano.function mapping X -> Kinv

        """
        K = self.kernel_.build_result(X, X)
        K += self.nu_ * T.identity_like(K)
        Kinv = T.nlinalg.matrix_inverse(K)
        Kinvfunc = theano.function([X], Kinv)
        return Kinvfunc

    def _build_Ksub_func(self, X1, X2):
        """Build function that calcuates kernel submatrix

        This method builds a thenao function that constructs a
        submatrix of the full kernel matrix given two batches of input
        data. This is also used to build the matrix of kernel evaluations
        between novel points and database points

        Args:
            X1 (theano.TensorType): symbolic tensor of first batch of input data
            X2 (theano.TensorType): symbolic tensor of second batch input data

        Returns:
            theano.function mapping (X1, X2) -> Ksub

        """
        Ksub = self.kernel_.build_result(X1, X2)
        Ksubfunc = theano.function([X1, X2], Ksub)
        return Ksubfunc

    @staticmethod
    def _build_inv_func(K):
        """Build function that inverts the kernel matrix

        This method takes a symbolic expression representing the kernel matrix
        and returns a theano function that  inverts it. This is used when K is
        constructed in batches and thus K cannot be built and inverted with the
        same function (as with Kinvfunc_)

        Args:
            K (theano dmatrix): symoblic kernel matrix

        Returns:
            theano.function mapping K -> Kinv

        """
        Kinv = T.nlinalg.matrix_inverse(K)
        invfunc = theano.function([K], Kinv)
        return invfunc

    def _build_add_nu_func(self, K):
        """Build function that adds nu to diagonal of kernel matrix

        This is used when K is constructed in batches and thus nu
        must be added after the matrix is complete (to avoid adding
        constant noise to off-diagonal terms).

        Args:
            K (theano dmatrix): symoblic kernel matrix

        Returns:
            theano.function mapping K -> K + nu*I

        """
        Knu = K + self.nu_ * T.identity_like(K)
        addnufunc = theano.function([K], Knu)
        return addnufunc

    def _build_pred_func(self, X, Y, Kinv, pred_std=False):
        """Build theano function for making predictions

        This methods builds the theano function that performs predictions given
        the database, novel points and the inverse kernel matrix from the database.
        This is done in two steps. First the matrix Kstar of kernel evaluations
        between novel points and database points is constructed. Then the vector
        of predictions is constructed with the equations described in the class
        docstring.

        Args:
            X (theano.TensorType): symbolic variable representing database input points
            Y (theano.dcol): symbolic vector of database output points
            Kinv (theano.dmatrix): symbolic inverse kernel matrix
            pred_std (bool): if True, output function will predict variance as well as mean

        Returns:
            theano.function mapping (Xstar, X, Y, Kinv) -> prediction

        """
        Xstar = self._get_x_var('Xstar')
        Kstar = self.kernel_.build_result(Xstar, X)
        mu = T.dot(T.dot(Kstar, Kinv), Y)
        if pred_std:
            sig = self.kernel_.build_result(Xstar, Xstar) - T.dot(kstar, T.dot(Kinv, Kstar.T))
            predfunc = theano.function([Xstar, X, Y, Kinv], [mu, sig])
        else:
            predfunc = theano.function([Xstar, X, Y, Kinv], mu)
        return predfunc

    @staticmethod
    def _build_batch_pred_func(Kstar, Y, Kinv):
        """Build prediction function for when Kstar is built in batches

        If there are too many database points to construct Kstar in one
        call of kernel.build_result, one cannot use predfunc_ because it assumes
        that one call will be sufficient. So Kstar must be built in batches and then
        input into the theano function that is output by this method. Note
        that prediction of variance is not implemented in the batch methods

        Args:
            Kstar (theano.dmatrix): symobolic matrix of kernel evaluations between novel and database points
            Y (theano.dcol): symbolic vector of database outputs
            Kinv (theano.dmatrix): symbolic inverse kernel matrix

        Returns:
            theano.function mapping (Kstar, Y, Kinv) -> prediction

        """
        mu = T.dot(T.dot(Kstar, Kinv), Y)
        batchpredfunc = theano.function([Kstar, Y, Kinv], mu)
        return batchpredfunc

    def _build_opt_step_func(self):
        """Build theano function that performs an optimzation step

        This methods constructs a theano function that first calculates the log
        likelihood of database points, then determines the gradient w.r.t to
        relevant parameters, and updates those parameters according to
                    params += learn_rate * grad(ll)

        Returns:
            theano.function that performs on step of stochastic gradient descent

        """
        X_batch = self._get_x_var('X_batch')
        Y_batch = T.dmatrix('Y_batch')

        # calculate K and inverse K
        K_batch = self.kernel_.build_result(X_batch, X_batch)
        K_batch += self.nu_ * T.identity_like(K_batch)
        Kinv_batch = T.nlinalg.matrix_inverse(K_batch)

        # calcualte log likelihood of data
        ll = -0.5 * T.dot(Y_batch.T, T.dot(Kinv_batch, Y_batch))
        ll -= 0.5 * (1. / T.nlinalg.det(Kinv_batch)) - 0.5 * X_batch.shape[0] * T.log(2 * np.pi)
        ll = ll.take(0)

        # find gradient wrt to parameters
        ll_grad = T.grad(ll, wrt=self.params_)

        updated = [self.params_[i] + self.learnRate_ * ll_grad[i] for i in range(len(self.params_))]
        updates = [(self.params_[i], updated[i]) for i in range(len(self.params_))]
        opt_step = theano.function([X_batch, Y_batch], ll, updates=updates)

        return opt_step

    def _get_x_var(self, name):
        """Get symbolic X vector with the correct rank

        Args:
            name (string): name of variable

        Returns:
            theano.TensorType variable representing input data
        """
        xrank_val = self.xrank_.get_value()
        if xrank_val == 2:
            x = T.dmatrix(name)
        elif xrank_val == 3:
            x = T.dtensor3(name)
        elif xrank_val == 4:
            x = T.dtensor4(name)
        else:
            raise ValueError("No kernels implemented for input tensors of rank %i" % xrank_val)
        return x


# Examples of functions for batch construction of K and subsequent predictions

def batch_build_K(gp, X1, X2, split_size1, split_size2, ydim=3, eq=False):
    if not gp.batch_:
        raise ValueError("Batch functions must be built in gp to use batch_build_K")
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    K = np.zeros((N1 * 3, N2 * 3), dtype=np.float64)
    splits1 = [i * split_size1 for i in range(1, int(np.floor(N1 / split_size1)))]
    splits2 = [i * split_size2 for i in range(1, int(np.floor(N2 / split_size2)))]

    x1splits = np.split(X1, splits1, axis=0)
    x2splits = np.split(X2, splits2, axis=0)

    nsplits1 = len(x1splits)
    nsplits2 = len(x2splits)

    for i in range(nsplits1):
        if eq:
            jrange = range(i, nsplits2)
        else:
            jrange = range(nsplits2)
        for j in jrange:
            kij = gp.K0func_(x1splits[i], x2splits[j])
            rows = i * ydim * split_size1, i * ydim * split_size1 + kij.shape[0]
            cols = j * ydim * split_size2, j * ydim * split_size2 + kij.shape[1]
            print(rows, cols)
            K[rows[0]:rows[1], cols[0]:cols[1]] = kij
            if eq and i != j:
                K[cols[0]:cols[1], rows[0]:rows[1]] = kij.T
    return K


def batch_build_Kinv(gp, X, split_size, ydim=3):
    if not gp.batch_:
        raise ValueError("Batch functions must be built in gp to use batch_build_Kinv")
    K = batch_build_K(gp, X, X, split_size, split_size, eq=True)
    K = gp.addnufunc_(K)
    Kinv = np.linalg.pinv(K, rcond=1e-15)
    return Kinv


def batch_pred(gp, Xstar, X, Y, Kinv, split_size_star, split_size_db, ydim=3):
    if not gp.batch_:
        raise ValueError("Batch functions must be built in gp to use batch_pred")
    kstar = batch_build_K(gp, Xstar, X, split_size_star, split_size_db, ydim=ydim)
    pred = gp.batchPred_(kstar, Y, Kinv)
    return pred
