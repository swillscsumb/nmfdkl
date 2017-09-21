import warnings

import numpy as np

class NMFDKL(object):
    """Based on https://github.com/romi1502/NMF-matlab/blob/ by Romain
        Hennequin
    """
    _EPSILON = np.spacing(1)

    def __init__(
             self, matrix, factors, bases_size, bases=None, weights_size=1,
             weights=None
             ):
        """
        """
        self._factors = None
        # V
        self.matrix = matrix
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
            self.bases *= np.max(self.matrix)
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
            self.weights *= np.max(self.matrix)

        self.one = np.ones((self.rows, self.columns))

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
        """shift array n columns
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
        """shift array n rows
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
        """
        k = np.log(self._EPSILON/max)/t
        return np.fromfunction(lambda i: max*np.exp(k*i), (t,))

    def lambda_(self):
        """L = sum((W)(H))
        """
        lam = np.zeros((self.rows, self.columns))
        weights_cs = None
        for t in xrange(self.bases.shape[1]):
            if weights_cs is None:
                weights_cs = self.weights[:, 0, :]
            else:
                # right shift the prior array so we only have to zero one
                # column
                weights_cs = self.shift_column(weights_cs, 1)
            lam += np.dot(self.bases[:, t, :], weights_cs)
        return lam

    def v_over_lambda(self):
        """V/L
        """
        lam = self.lambda_()
        vol = self.matrix/(lam + self._EPSILON)
        return vol, lam

    def update_bases_weights_t(self, vol, lam):
        """
        """
        vol_cs = None
        weights_tmp = np.zeros((self.factors, self.columns))
        for t in xrange(self.bases.shape[1]):
            weights_cs = self.shift_column(self.weights[:, 0, :], t).T
            self.bases[:, t, :] *= (
                np.dot(vol, weights_cs)
                /(np.dot(self.one, weights_cs) + self._EPSILON))
            bases_t = self.bases[:, t, :].T
            if vol_cs is None:
                vol_cs = vol
            else:
                # left shift the prior array so we only have to zero one
                # column
                vol_cs = self.shift_column(vol_cs, -1)

            weights_tmp += (
                np.dot(bases_t, vol_cs)
                /(np.dot(bases_t, self.one) + self._EPSILON))
        self.weights[:, 0, :] *= weights_tmp

    def update_bases(self, vol):
        """W = W * (((V/L)(H))/(1(H)))
        """
        weights_cs = None
        for t in xrange(self.bases.shape[1]):
            if weights_cs is None:
                weights_cs = self.weights[:, 0, :].T
            else:
                # use shift_row since we're shifting the transpose
                weights_cs = self.shift_row(weights_cs, 1)
            self.bases[:, t, :] *= (
                np.dot(vol, weights_cs)
                /(np.dot(self.one, weights_cs) + self._EPSILON))

    def update_weights(self, vol):
        """H = H * (((W.T)(V/L))/((W)1))
        """
        vol_cs = None
        weights_tmp = np.zeros((self.factors, self.columns))
        for t in xrange(self.bases.shape[1]):
            if vol_cs is None:
                vol_cs = vol
            else:
                # left shift the prior array so we only have to zero one
                # column
                vol_cs = self.shift_column(vol_cs, -1)
            bases_t = self.bases[:, t, :].T
            weights_tmp += (
                np.dot(bases_t, vol_cs)
                /(np.dot(bases_t, self.one) + self._EPSILON))

        self.weights[:, 0, :] *= weights_tmp

    def update_weights_avg(self, vol):
        """
        average along t
        """
        hu = np.zeros((self.factors, self.columns))
        hd = np.zeros((self.factors, self.columns))
        for r in xrange(self.factors):
            for m in xrange(self.rows):
                bases_flip = np.flipud(self.bases[m, :, r])
                cv = np.convolve(vol[m, :], bases_flip, 'same')
                hu[r, :] += cv
                cv = np.convolve(self.one[m, :], bases_flip, 'same')
                hd[r, :] += cv

        # average along t
        self.weights[:, 0, :] *= hu/hd

    def get_cost(self, vol, lam):
        """
        D = ||V * ln(V/L) - V + L||
                                   F
        """
        cost = np.linalg.norm(
            self.matrix*np.log(vol + self._EPSILON) - self.matrix + lam, 'fro')
        return cost

    def nmfdkl_iter(self, iterations):
        """Dittmar and Muller iteration algoritm
        """
        for i in xrange(iterations):
            warnings.warn("iteration: {0:d}".format(i+1))
            vol, lam = self.v_over_lambda()
            cost = self.get_cost(vol, lam)
            warnings.warn("cost: {0}".format(str(cost)))
            self.update_weights_avg(vol)
            vol, lam = self.v_over_lambda()
            self.update_bases(vol)

    def nmfdkl_dm_iter(self, iterations):
        """Dittmar and Muller iteration algoritm
        """
        for i in xrange(iterations):
            warnings.warn("iteration: {0:d}".format(i+1))
            vol, lam = self.v_over_lambda()
            cost = self.get_cost(vol, lam)
            warnings.warn("cost: {0}".format(str(cost)))
            self.update_bases_weights_t(vol, lam)

    def post_process_rh(self):
        """Post processing from NMF-matlab Romain Hennequin
        """
        weights_max = np.max(self.weights[:, 0, :])/10.0
        self.weights[:, 0, :] = (
            np.maximum(self.weights[:, 0, :], weights_max)
            - weights_max + self._EPSILON)

    def post_process(self):
        """Insert exponential decay envelopes in weights
        """
        d_env = self.decay(np.max(self.weights[:, 0, :]), self.bases.shape[1])
        for r in xrange(self.factors):
            self.weights[r, 0, :] = np.convolve(
                self.weights[r, 0, :], d_env, 'same')

    def reconstruct(self):
        """
        """
        v_out = np.zeros((self.rows, self.columns, self.factors+1))
        for r in xrange(self.factors):
            for m in xrange(self.rows):
                v_out[m, :, r] += np.convolve(
                    self.weights[r, 0, :], self.bases[m, :, r], 'same')
                v_out[m, :, -1] += v_out[m, :, r]
        return v_out


    def nmfdkl(self, pre, post):
        """
        """
        self.nmfdkl_iter(pre)
        self.post_process_rh()
        self.nmfdkl_iter(post)
        return self.reconstruct()
