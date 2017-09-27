import theano.tensor as T
import theano
import theano.tensor.slinalg
import numpy as np

__author__ = 'David Brookes'
__date__ = '8/16/17'

"""
This module contains classes that implement various kernel functions
"""


class BaseKernel(object):
    """Abstract base class for kernel implementations in theano.

    A kernel function by definition, computes the inner product
    of two data points in a transformed space. Each class
    is defined by some underlying kernel function, but is designed
    to operate on complete data sets, not individual pairs of
    data points. Thus, the output is the complete kernel matrix made
    up of evaluations of the kernel function on every pair of data
    points in the input data sets. That is, if the input data
    sets are X1=[x11, ... x1N] and X2=[x21,...x2N], and the
    underlying kernel function is k(x, x'), then the output
    from these classes is

            K(X1, X2) = [[k(x11, x21),...k(x1N, x21)]
                          ,...,      ,...,       ,...,
                         [k(x11, x2M),...,k(x1N, x2M)]]

    This is implemented in the 'build_result' function of every class.
    This matrix represents the covariance matrix of a Gaussian Process
    defined on the input space. If this Gaussian Process is scalar-valued
    (the usual case), then the kernel function will also be scalar-valued.
    However, if the GP is vector-valued, with dimensionality d, then the
    kernel function is matrix-valued (where each output is shape d x d)
    and the total kernel matrix will be a block matrix of size Nd x Md.

    Additionally, 'build_result' constructs a symbolic
    expression for the above equation, within the theano framework.
    Thus build_result takes as input two symbolic theano tensors and
    any required parameters are theano shared variables.

    This base class simply enforces the existence of 'build_result'
    and a 'get_params' function, which should return the relevant
    parameters for the kernel. This class is truly abstract, and will
    raise a NotImplementedError if one attempts to instantiate it.

    The initializing argument x_rank and associated member variable xrank_
    simply sets the rank of the tensors that are input to build_result.
    This is mostly used as a check for classes that use this kernel.
    It also contains an optional argument, ydim, that sets the dimensionality
    of the Gaussian Process this kernel describes. This is usually 1 (for a
    scalar-valued GP), but in some cases can be greater than 1.


    Args:
        x_rank (int): rank of the input tensor
        ydim (int): dimensionality of the GP this kernel models

    Attributes:
        xrank_ (theano shared int): rank of the input tensor
        ydim_ (int): dimensionality of the GP this kernel models

    """

    def __init__(self, x_rank, ydim=1.):
        self.xrank_ = theano.shared(x_rank)
        self.ydim_ = ydim

    def build_result(self, X1, X2):
        """Build symbolic expression for kernel evaluation

        This is the key function for these classes. It takes as input
        two data sets (possibly identical) and returns the kernel
        matrix made up of the evaluation of the kernel function on
        each pair of points in the input data sets.

        This function is not implemented in this base class and raises an
        exception if one attempts to instantiate the class

        Args:
            X1 (theano.TensorType): first data set
            X2 (theano.TensorType): second data set

        Returns:
            theano.TensorType variable representing the kernel matrix
            (if implemented)

        Raises:
            NotImplementedError: enforces that this method must be
            implemented by sub-classes, and thus that this base class
            cannot be instantiated

        """
        raise NotImplementedError

    def get_params(self):
        """Get the parameters for this kernel function

        This is a helper function that returns all of the
        relevant parameters for this kernel function. By parameters,
        we mean any quantities that are not directly determined
        by the input data. These should be theano shared variables
        that are members of the class and are set at initialization
        (and possibly updated later through optimization).

        This function is not implemented in this base class and raises an
        exception if one attempts to instantiate the class

        Returns:
            List of theano shared variables representing the
            relevant parameters for the kernel

        Raises:
            NotImplementedError: enforces that this method must be
            implemented by sub-classes, and thus that this base class
            cannot be instantiated

        """
        raise NotImplementedError


class BaseVectorKernel(BaseKernel):
    """ABC for kernels that operate on vector data points

    This is a base class for implementing any kernels whose underlying
    function operates on vector quantities. Thus, in all sub-classes of
    this type, the input data sets to 'build_result' should be matrices
    (rank 2 tensors), where each row is a data point.

    This base class simply implements a static method that computes the
    dot product between every pair of data points in the input
    data sets. This operation is useful for calculating most
    kernels that operate on vectors. As with the parent class, this
    class is truly abstract, and will raise a NotImplementedError
    if one attempts to instantiate it.

    """

    def __init__(self):
        super(BaseVectorKernel, self).__init__(2)

    @staticmethod
    def _pair_dot(X1, X2):
        """Calculate pairwise dot product

        This is convenience function for computing the dot product
        between all pairs of data points in the input data sets.

        Args:
            X1 (theano.TensorType): first data set matrix (each data point is a vector)
            X2 (theano.TensorType): second data set matrix (each data point is a vector)

        Returns:
            A theano.TensorType variable representing the matrix
            of dot product evaluations for each point in X1 and X2.
            If X1 and X2 have N and M points, respectively, then
            the output matrix will be NxM.

        """
        return T.dot(X1, X2.T)


class PolynomialKernel(BaseVectorKernel):
    """Implementation of the polynomial kernel function.

    This is a subclass of BaseVectorKernel that uses the polynomial
    kernel as its underlying function. The polynomial kernel function is:

            k(x, x') = (alpha*(x.x') + c)^d

    where alpha, c, and d are parameters.

    Args:
        alpha (float): scaling parameter
        c (float): parameter that sets the origin
        d (int): parameter setting the degree of polynomial

    Attributes:
        alpha_ (theano shared variable): scaling parameter
        c (theano shared variable): parameter that sets the origin
        d (theano shared variable): parameter setting the degree of polynomial

    """

    def __init__(self, d, alpha, c):
        self.d_ = theano.shared(d, name='d')
        self.alpha_ = theano.shared(alpha, name='alpha')
        self.c_ = theano.shared(c, name='c')
        super(PolynomialKernel, self).__init__()

    def get_params(self):
        """Retrieve the parameters

        Returns:
            List of theano shared variables: [d_, alpha_ and c_]

        """
        return [self.d_, self.alpha_, self.c_]

    def build_result(self, X1, X2):
        """Evaluate the polynomial kernel pairwise

        This symbolically evaluates the polynomial kernel on every pair of
        data points in the input data sets by first computing the pairwise
        dot product matrix with '_pair_dot', and then evaluating
        the polynomial expression on each element.

        Args:
            X1 (theano.TensorType): first data set matrix
            X2 (theano.TensorType): second data set matrix

        Returns:
            A theano.TensorType variable representing the kernel
            matrix with an underlying polynomial kernel function

        """
        X12 = self._pair_dXt(X1, X2)
        K = (self.alpha_ * X12 + self.c_) ** self.d_
        return K


class LinearKernel(PolynomialKernel):
    """Implement linear kernel

    The linear kernel is simply a special case of the polynomial
    kernel with alpha = d = 1. Thus the underlying kernel is:
            k(x, x') = x.x' + c

    Args:
        c (float): parameter that sets the origin

    """

    def __init__(self, c):
        super(LinearKernel, self).__init__(1, 1, c)

    def get_params(self):
        """Only return c_

        This is the adjustable parameter for this kernel

        Returns:
            The theano shared variable c_

        """
        return self.c_


class BaseStationaryVectorKernel(BaseVectorKernel):
    """ABC for stationary kernel functions

    This is the base class for any BaseVectorKernel subclasses
    that have stationary underlying functions, i.e., underlying
    functions that operate only on the distances between input
    data points.

    This base class simply implements a  method that computes the
    squard distance between every pair of data points in the input
    data sets. These are the inputs into any stationary kernel
    function. As with the parent class, this class is truly abstract,
    and will raise a NotImplementedError if one attempts to instantiate it.

    """

    def __init__(self):
        super(BaseStationaryVectorKernel, self).__init__()

    def _pair_sq_dist(self, X1, X2):
        """Calculate pairwise squared distance

        This is convenience function for computing the squared distance between
        all pairs of data points in the input data sets. This implementation is
        based on the observation that

                ||x-x||^2 = x.x + x'.x' - 2*x.x'

        See https://chrisjmccormick.wordpress.com/2014/08/22/fast-euclidean-distance-calculation-with-matlab-code/
        for more details.

        Args:
            X1 (theano.TensorType): first data set matrix
            X2 (theano.TensorType): second data set matrix

        Returns:
            A theano.TensorType variable representing the matrix
            of squared distances between each point in X1 and X2.
            If X1 and X2 have N and M points, respectively, then
            the output matrix will be NxM.

        """
        X11 = T.sum(X1 ** 2, axis=1).reshape((X1.shape[0], 1))
        X22 = T.sum(X2 ** 2, axis=1).reshape((1, X2.shape[0]))
        X12 = self._pair_dot(X1, X2)
        D = X11 + X22 - 2 * X12
        return D


class SEVectorKernel(BaseStationaryVectorKernel):
    """Implement squared exponential kernel

    This class uses a stationary underlying kernel function
    with the form:

            k(x, x') = alpha*exp(-(||x-x'||^2)/l^2)

    where alpha and l are parameters. This is known as the Squared
    Exponential (SE)  kernel and is broadly applicable.

    Args:
        alpha (float): parameter that sets the maximum value of the kernel
        l (float): sets the length scale of the kernel

    Attributes:
        alpha_ (theano shared variable): parameter that sets the maximum value of the kernel
        l (theano shared variable): parameter that sets the length scale of the kernel

    """

    def __init__(self, alpha, l):
        self.alpha_ = theano.shared(alpha, name='alpha')
        self.l_ = theano.shared(l, name='l')
        super(SEVectorKernel, self).__init__()

    def get_params(self):
        """Retrieve the parameters

        Returns:
            List of theano shared variables: [alpha_, l_]

        """
        return [self.alpha_, self.l_]

    def build_result(self, X1, X2):
        """Evaluate SE kernel pairwise

        This symbolically evaluates the SE kernel on every pair of
        data points in the input data sets by first computing the pairwise
        distance matrix with '_pair_sq_dist', and then evaluating
        the SE expression on each element.

        Args:
            X1 (theano.TensorType): first data set matrix
            X2 (theano.TensorType): second data set matrix

        Returns:
            A theano.TensorType variable representing the kernel
            matrix with an underlying SE kernel function

        """
        D = self._pair_sq_dist(X1, X2)
        K = self.alpha_ * T.exp(-D ** 2 / self.l_ ** 2)
        return K


class BaseEnvKernel(BaseKernel):
    """ABC for kernels that operate on chemical environments

    Chemical environments are well described by rank 3 tensors with
    shape nZ x M x d, where nZ is the number of unique chemical species,
    M is the maximum number of particles of each species in the environment
    (note that this is a different meaning for M then that used in the
    BaseKernel docstring) and d is the dimensionality of the space the
    particles are in (almost always 3). The input into these kernels is
    thus a rank 4 tensor containing N chemical environments.

    These kernels are all based on functional representations of
    environments, where each is described as a density field, p(r), of
    Gaussian functions centered on each particle position:

                p(r) = sum_z^nZ sum_i^M G(r_z_i; sig_z)

    where G(r;sig) is a Gaussian with center r and width sig, r_z_i is the
    position of the ith particle of species z and sig_z is the width of
    gaussians for species z. The overlap integral between two environments,
    S(p, p') = int dr p(r) p'(r), then serves as a natural measure of similarity,
    or base kernel. We can then give the kernel the desirable property of
    rotational invariance or covariance by integrating over all rotations, R,
    of the environments:

                    k(p, p') = int dR | S(p, Rp')|^q

    where the exponent q determines how much angular information is preserved.
    Subclasses of this class that are name 'Linear' have q=1 and 'Quadratic'
    have q=2 (coming soon!). The necessary properties and methods of derivation
    for these kernels are discussed extensively in [1], [2] and [3] below.

    [1] A.P. Bartok, R. Kondor and G. Csanyi. Phys. Rev. B. 87 (2013)
    [2] A. Glielmo, P. Sollich, and A. De Vita. Phys. Rev. B. 95 (2017)
    [3] G. Ferre, J.B. Maillet, and G. Stoltz. J. Chem. Phys. 143 (2015)

    This base class simply ensures that the input tensors are of rank 4 and that
    there are parameters setting the width of the Gaussian functions for each
    species.

    It also contains a method that implements a number of cutoff
    functions so atoms very far away from the origin in are
    not included in the kernel calculation, which is often desirable.
    The default is no cutoff, but if the input variable 'rcut' is not None
    then there will be a cutoff implemented at that rcut. The type
    of cutoff can be adjusted by the 'cut_style' input string. See
    _build_cutoff() for more details.

    Args:
        sigs (vector of floats): Gaussian widths for each species
        ydim (int): dimensionality of the GP that this kernel models
        rcut (float): cutoff distance. If None, no cutoff is implemented (default)
        cut_style (string): type of cutoff function ('behler' or 'step')
        opt_rcut (bool): if True, include cutoff radius in optimizable parameters

    Attributes:
        sigs_ (theano shared variable): Gaussian widths for each species
        rcut_ (theano shared variable or None): cutoff distance
        cutstyle_ (string): type of cutofff function
        optRcut (bool): if True, include cutoff radius in optimizable parameters

    """

    def __init__(self, sigs, ydim, rcut=None, cut_style='step', opt_rcut=False):
        self.sigs_ = theano.shared(sigs)
        self.optRcut_ = opt_rcut
        if rcut is None:
            self.rcut_ = rcut
        else:
            self.rcut_ = theano.shared(rcut)
        self.cutstyle_ = cut_style
        super(BaseEnvKernel, self).__init__(4, ydim=ydim)

    def cutoff(self, Xnorms):
        """Build a theano function implementing a cutoff

        The cutoff function acts on the norms of the position vectors in the
        input environment. Currently two options for cutoff functions are available,
        The default is 'step', which implements the basic step function

                    h(r) = 1 if r > rcut; 0 otherwise

        The second is 'behler' which implements a cutoff introduced in [1].
        It has the form:

            h(r) = 0.5 * (cos(pi * r/rcut) + 1) if r >  rcut; 0 otherwise

        This provides a much smoother cutoff than the step function

        [1] J. Behler and M. Parrinello. Phys. Rev. Lett. 98 (2007)

        Args:
            Xnorms: theano tensor representing the norms of an environment's position vectors

        Returns:
            theano tensor of elementwise evaluations of the cutoff function on the
            input distances.

        """
        rcut_mat = self.rcut_ * T.ones_like(Xnorms)
        cond = T.lt(Xnorms, rcut_mat) * T.neq(Xnorms, T.zeros_like(Xnorms))  # tensor of booleans (elementwise X < rcut)
        if self.cutstyle_ == 'step':
            iftrue = Xnorms
        elif self.cutstyle_ == 'behler':
            iftrue = 0.5 * (T.cos(np.pi * Xnorms / self.rcut_) + 1)
        else:
            raise ValueError("Cutoff function \'%s\' not implemented" % style)

        cuts = T.switch(cond, iftrue, T.zeros_like(Xnorms))
        return cuts

    def get_params(self):
        """Retrieve the parameters

        Returns:
            List with the theano shared variables sigs_ and rcut_ (if not None)

        """
        if (self.rcut_ is not None) and (self.optRcut_ is True):
            return [self.sigs_, self.rcut_]
        else:
            return [self.sigs_]


class LinearScalarEnvKernel(BaseEnvKernel):
    """Linear environment kernel for predicting scalar quantities

    This class implements a kernel on chemical environments that is linear
    (in the sense described by in the base class docstring) and can be
    used to describe the covariance between points of a scalar-valued
    Gaussian Process. This kernel is invariant to rotations, translations
    and permutations of particles of the same species. The underlying
    kernel function between two environments is given by

     k(p, p') = sum_z^nZ (Cz) sum_i^M sum_j^M exp(-alpha_ij) * (sinh(gamma_ij)/gamma_ij))

    where alpha_ij = (ri^2+rj'^2)/4*sig_z**2, gamma_ij = (ri*rj')/2*sig_z**2 and
    Cz = 1/(8*(pi*sig_z**2)^(3/2)), and ri and rj' are the position magnitudes
    of the ith particle in p and the jth particle in p' of type z, respectively.

    Args:
        sigs (vector of floats): Gaussian widths for each species

    """

    def __init__(self, sigs, rcut=None, cut_style='step'):
        super(LinearScalarEnvKernel, self).__init__(sigs, 1, rcut=rcut,
                                                    cut_style=cut_style)

    def build_result(self, X1, X2):
        """Theano implementation of this kernel

        The major steps in the evalution of this kernel is calculating
        the rank 5 tensors (w/ shape nZ x N1 x N2 x M1 x M2) containing
        gamma_ij and alpha_ij between every pair of particle position magnitudes
        of the same species in every pair of environments. After these are constructed,
        we perform elementwise operations to calculate the individual components
        of the sum described above, and then sum over the nZ, M1 and M2
        dimensions to calculate the final kernel

        Args:
            X1 (theano.TensorType): rank 4 tensor containing a set of environmXnts
            X2 (theano.TensorType): rank 4 tensor containing a set of environments

        Returns:
            A theano.TensorType variable representing the resulting kernel matrix

        """
        X1 = X1.dimshuffle(1, 0, 2, 3)  # easier to deal with if the species axis Xs fiXst
        X2 = X2.dimshuffle(1, 0, 2, 3)

        sigs = self.sigs_.dimshuffle(0, 'x', 'x', 'x', 'x')  # to ensure proper broadcasting
        Cz = 1. / ((2 * T.sqrt(np.pi * sigs ** 2)) ** 3)

        #  Calc distance magnitudes and resize to ensure broadcasting that
        #  perform calculations between every pair of particles of the same species
        #  in every pair of environments
        x1 = X1.norm(2, axis=-1).dimshuffle(0, 1, 'x', 2, 'x')
        x2 = X2.norm(2, axis=-1).dimshuffle(0, 'x', 1, 'x', 2)

        Gamma = (x1 * x2) / (2 * sigs ** 2)  # rank 5 tensor of gamma_ij values
        Alpha = (x1 ** 2 + x2 ** 2) / (4 * sigs ** 2)  # rank 5 tensor of alpha_ij values

        K = Cz * T.exp(Gamma - Alpha) + T.exp(-Alpha - Gamma)  # rank 5 tensor of individual components of kernel

        # multiply by elementwise cutoff if applicable
        if self.rcut_ is not None:
            H = self.cutoff(x1) * self.cutoff(x2)
            K *= H

        K *= 1 / (2 * Gamma)
        K = K.sum(axis=(0, -2, -1))

        return K


class LinearVectorEnvKernel(BaseEnvKernel):
    """Linear environment kernel for predicting vector quantites

    This class implements a kernel on chemical environments that is linear
    (in the sense described by in the base class docstring) and can be
    used to describe the covariance between points of a vector-valued
    Gaussian Process whose dimensionality is the same as that of the
    space the particles exist in (i.e. almost always 3). Because the GP is
    vector-valued, the kernel function is matrix-valued and the complete kernel
    matrix is a block matrix of shape (Nd x Nd) where d is the dimensionality of
    the GP and particle space (almost always 3).

    This kernel is invariant to translations and permutations
    of particles of the same species and is COvariant to rotations of the input
    environments. This means that if a rotation matrix acts on an environment,
    then the kernel evaluations on that environment will
    also be acted on by that rotation matrix. These considerations
    and the derivation of this kernel are discussed extensively in [2].
    Additionally, the form of the kernel is given in Eq. (30) of [2].

    [2] A. Glielmo, P. Sollich, and A. De Vita. Phys. Rev. B. 95 (2017)

    Args:
        sigs (vector of floats): Gaussian widths for each species

    """

    def __init__(self, sigs, rcut=None, cut_style='step'):
        super(LinearVectorEnvKernel, self).__init__(sigs, 3, rcut=rcut,
                                                    cut_style=cut_style)

    def build_result(self, X1, X2):
        """Theano implementation of this kernel

        The implemenentation of this kernel is very similar to that of the
        linear kernel for scalar predictions, with the additional step of creating
        the rank 7 tensor (of shape nZ x N1 x N2 x M1 x M2 x d x d) between
        every pair of particle position vectors (scaled to be unit length) of
        the same species in every pair of environments. These are then scaled
        by a rank 5 tensor (reshaped to rank 7 to ensure proper broadcasting) that is
        similar to the linear kernel for scalar predictions. We then sum over the
        nZ, M1 and M2 dimensions to get to the final block kernel matrix.

        Args:
            X1 (theano.TensorType): rank 4 tensor containing a set of environmXnts
            X2 (theano.TensorType): rank 4 tensor containing a set of environments

        Returns:
            A theano.TensorType variable representing the resulting block kernel matrix

        """
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        X1 = X1.dimshuffle(1, 0, 2, 3)
        X2 = X2.dimshuffle(1, 0, 2, 3)
        sigs = self.sigs_.dimshuffle(0, 'x', 'x', 'x', 'x')
        Cz = 1. / ((2 * T.sqrt(np.pi * sigs ** 2)) ** 3)

        x1 = X1.norm(2, axis=-1)
        x2 = X2.norm(2, axis=-1)

        # Make unit position vectors and reshape for pairwise broadcasting
        X1_unit = X1 / x1.dimshuffle(0, 1, 2, 'x')
        X2_unit = X2 / x2.dimshuffle(0, 1, 2, 'x')
        X1_unit = X1_unit.dimshuffle(0, 1, 'x', 2, 'x', 3, 'x')
        X2_unit = X2_unit.dimshuffle(0, 'x', 1, 'x', 2, 'x', 3)
        OP = X1_unit * X2_unit  # calculate all outer products

        # calculate scaling factor for the outer products (phi)
        x1 = x1.dimshuffle(0, 1, 'x', 2, 'x')
        x2 = x2.dimshuffle(0, 'x', 1, 'x', 2)
        Gamma = (x1 * x2) / (2 * sigs ** 2)
        Alpha = (x1 ** 2 + x2 ** 2) / (4 * sigs ** 2)

        Phi = Cz * ((Gamma - 1) * T.exp(Gamma - Alpha) + (Gamma + 1) * T.exp(-Gamma - Alpha)) / (Gamma ** 2) / 2

        # multiply by elementwise cutoff if applicable
        if self.rcut_ is not None:
            H = self.cutoff(x1) * self.cutoff(x2)
            Phi *= H

        Phi = Phi.dimshuffle(0, 1, 2, 3, 4, 'x', 'x')  # scaled Phi so it can broadcast with OP

        K = Phi * OP  # rank 7 tensor containing each component of the kernel

        # sum and reshape to get the final rank 2 tensor
        K = K.sum(axis=(0, 3, 4))
        K = K.dimshuffle(0, 2, 1, 3)
        K = K.reshape((N1 * self.ydim_, N2 * self.ydim_))
        return K


class QuadraticVectorEnvKernel(BaseEnvKernel):
    """Quadratic environment kernel for predicting vector quantites

    This class implements a kernel on chemical environments that is quadratic
    (in the sense described by in the base class docstring) and can be
    used to describe the covariance between points of a vector-valued
    Gaussian Process whose dimensionality is the same as that of the
    space the particles exist in (i.e. almost always 3). Because the GP is
    vector-valued, the kernel function is matrix-valued and the complete kernel
    matrix is a block matrix of shape (Nd x Nd) where d is the dimensionality of
    the GP and particle space (almost always 3).

    This kernel is invariant to translations and permutations
    of particles of the same species and is COvariant to rotations of the input
    environments. The derivation of this kernel is a simple extension of the ideas 
    in [2].  

    [2] A. Glielmo, P. Sollich, and A. De Vita. Phys. Rev. B. 95 (2017)

    Args:
        sigs (vector of floats): Gaussian widths for each species

    """

    def __init__(self, sigs):
        super(QuadraticVectorEnvKernel, self).__init__(sigs, 3)

    def build_result(self, X1, X2):
        """Theano implementation of this kernel

        The implemenentation of this kernel is very similar to that of the
        linear kernel for quadratic predictions but we must now consider pairs of 
        particle position vectors.  Specifically, we need the sum of square magnitudes
        of possible pairs of position vectors in the set for each environment
        (rp) as well as the vector sum of all possible pairs of position vectors in 
        the set for each environemnt (Rq). The outer product of the latter quantities
        (Rq1*Rq2) is a rank 10 tensor of shape nZ x nZ x N1 x N2 x M1 x M1 x M2
        x M2 x d x d.  This is then scaled by a rank 8 tensor (reshaped to rank 10 
        to ensure proper broadcasting) that is a function of rp and rq (the norm of Rq)
        We then sum over the nZ, nZ, M1, M1, M2, and M2 dimensions to get to the 
        final block kernel matrix.

        Args:
            X1 (theano.TensorType): rank 4 tensor containing a set of environmXnts
            X2 (theano.TensorType): rank 4 tensor containing a set of environments

        Returns:
            A theano.TensorType variable representing the resulting block kernel matrix

        """

        N1, nZ, M1, d = X1.shape
        N2, M2 = X2.shape[0], X2.shape[0]

        X1 = X1.dimshuffle(1, 0, 2, 3)
        X2 = X2.dimshuffle(1, 0, 2, 3)
        sigs = self.sigs_.dimshuffle(0, 'x', 'x', 'x', 'x', 'x', 'x', 'x')
        Cz = 1. / (8 * (np.pi * sigs ** 2) ** (3 / 2)) ** 2

        # Make rp1 rp2
        x1 = X1.norm(2, axis=-1) ** 2
        x2 = X2.norm(2, axis=-1) ** 2
        rp1 = x1.dimshuffle(0, 'x', 1, 2, 'x')  # Add axes nZp, M1p
        rp2 = x2.dimshuffle(0, 'x', 1, 2, 'x')
        rp1 = rp1 + rp1.dimshuffle(1, 0, 2, 4, 3)  # Symmetrize of nZ x nZp and M1 x M1p
        # This gives sums of pairs of square magnitudes

        rp2 = rp2 + rp2.dimshuffle(1, 0, 2, 4, 3)

        # Make Rq1 Rq2 and norms/unit vectors
        Rq1 = X1.dimshuffle(0, 'x', 1, 2, 'x', 3)  # Add axes nZp, M1p
        Rq2 = X2.dimshuffle(0, 'x', 1, 2, 'x', 3)
        Rq1 = Rq1 + Rq1.dimshuffle(1, 0, 2, 4, 3, 5)  # Symetrize over nZ x nZp and M1 x M1p
        Rq2 = Rq2 + Rq2.dimshuffle(1, 0, 2, 4, 3, 5)
        rq1 = Rq1.norm(2, axis=-1)
        rq2 = Rq2.norm(2, axis=-1)
        Rqu1 = Rq1 / rq1.dimshuffle(0, 1, 2, 3, 4, 'x')
        Rqu2 = Rq2 / rq2.dimshuffle(0, 1, 2, 3, 4, 'x')

        # Make outer product
        OP = Rqu1.dimshuffle(0, 1, 2, 'x', 3, 4, 'x', 'x', 5, 'x') * Rqu2.dimshuffle(0, 1, 'x', 2, 'x', 'x', 3, 4, 'x',
                                                                                     5)
        # Make phi
        rq1 = rq1.dimshuffle(0, 1, 2, 'x', 3, 4, 'x', 'x')  # Add blank axes for N2, M2, M2p
        rq2 = rq2.dimshuffle(0, 1, 'x', 2, 'x', 'x', 3, 4)  # Add blank axes for N1, M1, M1p
        rp1 = rp1.dimshuffle(0, 1, 2, 'x', 3, 4, 'x', 'x')  # Add blank axes for N2, M2, M2p
        rp2 = rp2.dimshuffle(0, 1, 'x', 2, 'x', 'x', 3, 4)  # Add blank axes for N1, M1, M1p
        alpha = (rp1 + rp2) / (4 * sigs ** 2)
        gamma = (rq1 * rq2) / (2 * sigs ** 2)
        phi = Cz * (T.exp(-alpha) / gamma ** 2) * (gamma * T.cosh(gamma) - T.sinh(gamma))
        phi = phi.dimshuffle(0, 1, 2, 3, 4, 5, 6, 7, 'x', 'x')  # Adding blank axes d, dp for broadcasting with OP

        K = phi * OP  # rank 10 tensor containing each component of the kernel

        # sum and reshape to get the final rank 2 tensor
        K = K.sum(axis=(0, 1, 4, 5, 6, 7)).dimshuffle(0, 2, 1, 3)
        K = K.reshape((N1 * d, N2 * d))
        return K
