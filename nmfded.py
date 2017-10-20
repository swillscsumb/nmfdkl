import warnings

import numpy as np

class NMFDED(object):
    """
    Based on https://github.com/romi1502/NMF-matlab/ by Romain
    Hennequin

    Implements Non-negative Matrix Factor Deconvolution (NMFD) as
    proposed by Smaragdis.

    Uses Least Squares method from Schmidt and Morup and Wang,
    Cichocki, and Chambers.

    Schmidt, M. N. & Morup, M. (2006). Non-negative matrix factor 2-d
        deconvolution for blind single channel source separation. Independent
        Component Analysis, International Conference on (ICA), Springer
        Lecture Notes in Computer Science, Vol.3889, 700-707.
        Retrieved from http://mikkelschmidt.dk/papers/schmidt2006ica.pdf

    Smaragdis, P. (2004). Non-negative matrix factor deconvolution;
        extraction of multiple sound sources from monophonic inputs.
        Lecture Notes in Computer Science (including subseries Lecture
        Notes in Artificial Intelligence and Lecture Notes in
        Bioinformatics), 3195, 494-499. Retrieved from
        http://www.merl.com/reports/docs/TR2004-104.pdf

    Wang, W., Cichocki, A., & Chambers. J. (2009). A multiplicative algorithm
        for convolutive non-negative matrix factorization based on squared
        euclidean distance. IEEE Transactions on Signal Processing, 57, (7),
        2858-2864.

    """
    _EPSILON = np.spacing(1)
    _H_IDX = 0

    def __init__(
             self, matrix, factors, bases_size, bases=None, weights_size=1,
             weights=None, debug=True
             ):
        """
        nmfded.NMFDED
            matrix, factors, bases_size, bases=None, weights_size=1,
            weights=None)
        Parameters
        ----------
        matrix : array_like
            V MxN matrix to factor
        factors : int
            R decomposition components
        bases_size : int
            size of bases factor templates
        bases : array_like
            W initial MxTxR
        weights_size : int
            holding place for future NMF2D implementation.
            size of weights factor templates
        weights : array_like
            H initial RxTxN matrix
        Returns
        -------
        NMFD()
        """
        self._factors = None
        # V
        self.matrix_max = np.max(matrix)
        self.matrix_inv_max = 1.0 / self.matrix_max
        # normalize input to (0.0, 1.0]
        self.matrix = matrix * self.matrix_inv_max
        self.factors = factors
        # W
        if bases is not None:
            assert(
                matrix.shape[0] == bases.shape[0]
                and bases_size == bases.shape[1]
                and bases_size <= bases.shape[0]
                and factors == bases.shape[2])
            self.bases = bases.copy()
        else:
            self.bases = np.random.rand(
                self.rows, min(matrix.shape[0], bases_size), self.factors)
        # H
        if weights is not None:
            assert(
                factors == weights.shape[0]
                and weights_size == weights.shape[1]
                and matrix.shape[1] == weights.shape[2])
            self.weights = weights.copy()
        else:
            self.weights = np.random.rand(
                self.factors, weights_size, self.columns)

        self.one = np.ones((self.rows, self.columns))

        self._debug = debug

    @property
    def factors(self):
        return self._factors

    @factors.setter
    def factors(self, value):
        self._factors = value

    @property
    def rows(self):
        return self.matrix.shape[0]

    @property
    def columns(self):
        return self.matrix.shape[1]

    def shift_column(self, a, shift):
        """
        shift array n columns
        Parameters
        ----------
        a : array_like
            array to shift
        shift : int
            number of indexes to shift
        Returns
        -------
        out : array_like
            shifted array
        """
        if shift == 0:
            return a
        a_roll = np.roll(a, shift)
        if shift > 0:
            a_roll[:, :shift] = 0
        else:
            a_roll[:, shift:] = 0
        return a_roll

    def shift_row(self, a, shift):
        """
        shift array n rows
        Parameters
        ----------
        a : array_like
            array to shift
        shift : int
            number of indexes to shift
        Returns
        -------
        out : array_like
            shifted array
        """
        if shift == 0:
            return a
        a_roll = np.roll(a, shift, axis=0)
        if shift > 0:
            a_roll[:shift, :] = 0
        else:
            a_roll[shift:, :] = 0
        return a_roll

    def decay(self, max, t):
        """
        Create exponential decay envelope
        y = y0 * e ** (k*t)
        Parameters
        ----------
        max : float
            peak value of envelope
        t : int
            length of envelope
        Returns
        -------
        out : array_like
            t length array
        """
        k = np.log(self._EPSILON/max)/t
        return np.fromfunction(lambda i: max*np.exp(k*i), (t,))

    def lambda_(self):
        """
        L = sum((W)(H))
        """
        lam = np.zeros((self.rows, self.columns))

        for m in xrange(self.rows):
            for r in xrange(self.factors):
                lam[m, :] += np.convolve(
                    self.bases[m, :, r],
                    self.weights[r, self._H_IDX, :])[:self.columns]

        return lam

    def v_over_lambda(self):
        """
        V/L
        """
        lam = self.lambda_()
        vol = self.matrix/(lam + self._EPSILON)
        return vol, lam

    def update_bases(self, lam):
        """
        dot product bases update
        W = W * (V(H.T))/(L(H.T))
        """
        weights_cs = None
        for t in xrange(self.bases.shape[1]):
            if weights_cs is None:
                weights_cs = self.weights[:, self._H_IDX, :].T
            else:
                # use shift_row since we're shifting the transpose
                weights_cs = self.shift_row(weights_cs, 1)
            self.bases[:, t, :] *= (
                np.dot(self.matrix, weights_cs)
                /(np.dot(lam, weights_cs) + self._EPSILON))

    def update_weights(self, lam):
        """
        dot product weights update
        H = H * (((W.T)V)/((W.T)L))
        """
        v_cs = None
        lam_cs = None
        weights_num = np.zeros((self.factors, self.columns))
        weights_den = np.zeros((self.factors, self.columns))
        for t in xrange(self.bases.shape[1]):
            if v_cs is None and lam_cs is None:
                v_cs = self.matrix
                lam_cs = lam
            else:
                # left shift the prior array so we only have to zero one
                # column
                v_cs = self.shift_column(v_cs, -1)
                lam_cs = self.shift_column(lam_cs, -1)
            bases_t = self.bases[:, t, :].T
            weights_num += np.dot(bases_t, v_cs)
            weights_den += np.dot(bases_t, lam_cs)

        self.weights[:, self._H_IDX, :] *= weights_num/(weights_den + self._EPSILON)

    def update_weights_conv(self, lam):
        """
        convolution weights update
        H = H * (((W.T)V)/((W.T)L))
        """
        v_lr = np.fliplr(self.matrix)
        lam_lr = np.fliplr(lam)
        weights_num = np.zeros((self.factors, self.columns))
        weights_den = np.zeros((self.factors, self.columns))
        for r in xrange(self.factors):
            for m in xrange(self.rows):
                weights_num[r, :] += np.convolve(
                    v_lr[m, :], self.bases[m, :, r])[:self.columns]
                weights_den[r, :] += np.convolve(
                    lam_lr[m, :], self.bases[m, :, r])[:self.columns]
        weights_num = np.fliplr(weights_num)
        weights_den = np.fliplr(weights_den)
        self.weights[:, self._H_IDX, :] *= weights_num/(weights_den + self._EPSILON)

    def update_weights_avg(self, vol):
        """
        Hennequin convolution weights update with averaging
        """
        hu = np.zeros((self.factors, self.columns))
        hd = np.zeros((self.factors, self.columns))
        for r in xrange(self.factors):
            for m in xrange(self.rows):
                bases_flip = np.flipud(self.bases[m, :, r])
                hu[r, :] += np.convolve(vol[m, :], bases_flip)[:self.columns]
                hd[r, :] += np.convolve(
                    self.one[m, :], bases_flip)[:self.columns]

        # average along t
        self.weights[:, self._H_IDX, :] *= hu/hd

    def get_cost(self, lam):
        """
        Euclidean distance cost
        D = ||V - L||^2
        """
        dist = self.matrix - lam
        cost = np.linalg.norm(dist * dist, 'fro')
        return cost

    def nmfded_iter(self, iterations):
        """
        Hennequin iteration algoritm
        """
        for i in xrange(iterations):
            if self._debug:
                warnings.warn("iteration: {0:d}".format(i+1))
            lam = self.lambda_()
            cost = self.get_cost(lam)
            if self._debug:
                warnings.warn("cost: {0}".format(str(cost)))

            self.update_bases(lam)
            lam = self.lambda_()
            self.update_weights_avg(lam)

    def nmfded_dm_iter(self, iterations):
        """
        Dittmar and Muller like iteration algoritm
        """
        for i in xrange(iterations):
            if self._debug:
                warnings.warn("iteration: {0:d}".format(i+1))
            lam = self.lambda_()
            cost = self.get_cost(lam)
            if self._debug:
                warnings.warn("cost: {0}".format(str(cost)))
            self.update_bases(lam)
            lam = self.lambda_()
            self.update_weights_conv(lam)

    def post_process_rh(self):
        """
        Post processing from NMF-matlab Romain Hennequin
        """
        weights_max = np.max(self.weights[:, self._H_IDX, :])/10.0
        self.weights[:, self._H_IDX, :] = (
            np.maximum(self.weights[:, self._H_IDX, :], weights_max)
            - weights_max + self._EPSILON)

    def post_process(self):
        """
        Insert exponential decay envelopes in weights
        """
        d_env = self.decay(
            np.max(self.weights[:, self._H_IDX, :]), 2*self.bases.shape[1])
        for r in xrange(self.factors):
            self.weights[r, self._H_IDX, :] = np.convolve(
                self.weights[r, self._H_IDX, :], d_env)[:self.weights.shape[2]]

    def reconstruct(self):
        """
        reconstruct component factor matrices
        Parameters
        ----------

        Returns
        -------
        v_out : array_like
            MxNxR
        """
        v_out = np.zeros((self.rows, self.columns, self.factors+1))
        for r in xrange(self.factors):
            for m in xrange(self.rows):
                v_out[m, :, r] += np.convolve(
                    self.weights[r, self._H_IDX, :],
                    self.bases[m, :, r])[:self.columns]
                v_out[m, :, -1] += v_out[m, :, r]
        v_out *= self.matrix_max
        return v_out


    def nmfded(self, pre, post):
        """
        """
        self.nmfded_dm_iter(pre)
        self.post_process_rh()
        self.nmfded_dm_iter(post)
        return self.reconstruct()
