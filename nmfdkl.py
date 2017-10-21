import warnings

import numpy as np

class NMFDKL(object):
    """
    Based on https://github.com/romi1502/NMF-matlab/ by Romain
    Hennequin

    Implements Non-negative Matrix Factor Deconvolution (NMFD) as
    proposed by Smaragdis.

    Smaragdis, P. (2004). Non-negative matrix factor deconvolution;
        extraction of multiple sound sources from monophonic inputs.
        Lecture Notes in Computer Science (including subseries Lecture
        Notes in Artificial Intelligence and Lecture Notes in
        Bioinformatics), 3195, 494-499. Retrieved from
        http://www.merl.com/reports/docs/TR2004-104.pdf
    """
    _EPSILON = np.spacing(1)
    _H_IDX = 0

    def __init__(
             self, matrix, factors, bases_size, bases=None, weights_size=1,
             weights=None, debug=True
             ):
        """
        nmfdkl.NMFDKL(
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
                and factors == bases.shape[2])
            self.bases = bases.copy()
        else:
            self.bases = np.random.rand(self.rows, bases_size, self.factors)
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

    def update_bases(self, vol):
        """
        dot product bases update
        W = W * (((V/L)(H))/(1(H)))
        """
        weights_cs = None
        for t in xrange(self.bases.shape[1]):
            if weights_cs is None:
                weights_cs = self.weights[:, self._H_IDX, :].T
            else:
                # use shift_row since we're shifting the transpose
                weights_cs = self.shift_row(weights_cs, 1)
            self.bases[:, t, :] *= (
                np.dot(vol, weights_cs)
                /(np.dot(self.one, weights_cs) + self._EPSILON))

    def update_weights(self, vol):
        """
        dot product weights update
        H = H * (((W.T)(V/L))/((W)1))
        """
        vol_cs = None
        weights_num = np.zeros((self.factors, self.columns))
        weights_den = np.zeros((self.factors, self.columns))
        for t in xrange(self.bases.shape[1]):
            if vol_cs is None:
                vol_cs = vol
            else:
                # left shift the prior array so we only have to zero one
                # column
                vol_cs = self.shift_column(vol_cs, -1)
            bases_t = self.bases[:, t, :].T
            weights_num += np.dot(bases_t, vol_cs)
            weights_den += np.dot(bases_t, self.one)

        self.weights[:, self._H_IDX, :] *= weights_num/(weights_den + self._EPSILON)

    def update_weights_conv(self, vol):
        """
        convolution weights update
        H = H * (((W.T)(V/L))/((W)1))
        """
        vol_lr = np.fliplr(vol)
        weights_num = np.zeros((self.factors, self.columns))
        weights_den = np.zeros((self.factors, self.columns))
        for r in xrange(self.factors):
            for m in xrange(self.rows):
                weights_num[r, :] += np.convolve(
                    vol_lr[m, :], self.bases[m, :, r])[:self.columns]
                weights_den[r, 0] += self.bases[m, :, r].sum()
            weights_den[r, :] = weights_den[r, 0] + self._EPSILON
        weights_num = np.fliplr(weights_num)
        self.weights[:, self._H_IDX, :] *= weights_num/weights_den

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

    def get_cost(self, vol, lam):
        """
        Kullback-Leibler cost
        D = ||V * ln(V/L) - V + L||
                                   F
        """
        cost = np.linalg.norm(
            self.matrix*np.log(vol + self._EPSILON) - self.matrix + lam,
            'fro')
        return cost

    def nmfdkl_iter(self, iterations):
        """
        Hennequin iteration algoritm
        """
        for i in xrange(iterations):
            if self._debug:
                warnings.warn("iteration: {0:d}".format(i+1))
            vol, lam = self.v_over_lambda()
            cost = self.get_cost(vol, lam)
            if self._debug:
                warnings.warn("cost: {0}".format(str(cost)))

            self.update_bases(vol)
            vol, lam = self.v_over_lambda()
            self.update_weights_avg(vol)

    def nmfdkl_dm_iter(self, iterations):
        """
        Dittmar and Muller like iteration algoritm
        """
        for i in xrange(iterations):
            if self._debug:
                warnings.warn("iteration: {0:d}".format(i+1))
            vol, lam = self.v_over_lambda()
            cost = self.get_cost(vol, lam)
            if self._debug:
                warnings.warn("cost: {0}".format(str(cost)))
            self.update_bases(vol)
            vol, lam = self.v_over_lambda()
            self.update_weights_conv(vol)

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


    def nmfdkl(self, pre, post):
        """
        """
        self.nmfdkl_dm_iter(pre)
        self.post_process_rh()
        self.nmfdkl_dm_iter(post)
        return self.reconstruct()
