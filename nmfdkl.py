import warnings

import numpy as np


def shift_column(a, shift):
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

def shift_row(a, shift):
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

class NMFDKL(object):
    """
    """
    _EPSILON = np.spacing(1)

    def __init__(
             self, matrix, factors, bases_size, bases=None, weights_size=1,
             weights=None
             ):
        """
        """
        self._factors = None
        # v
        self.matrix = matrix
        self.factors = factors
        # w
        if bases is not None:
            assert(
                matrix.shape[0] == bases.shape[0]
                and bases_size == bases.shape[1]
                and factors == bases.shape[2])
            self.bases = bases
        else:
            self.bases = np.random.rand(self.rows, bases_size, self.factors)
        # h
        if weights is not None:
            assert(
                factors == weights.shape[0]
                and weights_size == weights.shape[1]
                and matrix.shape[1] == weights.shape[2])
            self.weights = weights
        else:
            self.weights = np.random.rand(
                self.factors, weights_size, self.columns)

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

    def lambda_(self):
        lam = np.zeros((self.rows, self.columns))
        weights_cs = None
        for t in xrange(self.bases.shape[1]):
            if weights_cs is None:
                weights_cs = self.weights[:, 0, :]
            else:
                # right shift the prior array so we only have to zero one
                # column
                weights_cs = shift_column(weights_cs, 1)
            lam += np.dot(self.bases[:, t, :], weights_cs)
        return lam

    def v_over_lambda(self):
        vol = self.matrix/(self.lambda_() + self._EPSILON)
        return vol

    def update_bases(self, vol):
        """
        """
        weights_cs = None
        for t in xrange(self.bases.shape[1]):
            if weights_cs is None:
                weights_cs = self.weights[:, 0, :].T
            else:
                # use shift_row since we're shifting the transpose
                weights_cs = shift_row(weights_cs, 1)
            self.bases[:, t, :] *= (
                np.dot(vol, weights_cs)
                /(np.dot(self.one, weights_cs) + self._EPSILON))

    def update_weights(self, vol):
        """
        """
        vol_cs = None
        weights_tmp = np.zeros((self.factors, self.columns))
        for t in xrange(self.bases.shape[1]):
            if vol_cs is None:
                vol_cs = vol
            else:
                # left shift the prior array so we only have to zero one
                # column
                vol_cs = shift_column(vol_cs, -1)
            bases_t = self.bases[:, t, :].T
            weights_tmp += (
                np.dot(bases_t, vol_cs)
                /(np.dot(bases_t, self.one) + self._EPSILON))

        self.weights[:, 0, :] *= weights_tmp

    def nmfdkl_iter(self, iterations):
        """
        """
        for i in xrange(iterations):
            warnings.warn("iteration: {0:d}".format(i+1))
            vol = self.v_over_lambda()
            self.update_weights(vol)
            vol = self.v_over_lambda()
            self.update_bases(vol)


    def post_process(self):
        weights_max = np.max(self.weights[:, 0, :])/10.0
        self.weights[:, 0, :] = (
            np.maximum(self.weights[:, 0, :], weights_max)
            - weights_max + self._EPSILON)

    def reconstruct(self):
        """
        """
        v_out = np.zeros((self.rows, self.columns, self.factors+1))
        for r in xrange(self.factors):
            for m in xrange(self.rows):
                cv = np.convolve(self.bases[m, :, r], self.weights[r, 0, :])
                v_out[m, :, r] += cv[:self.columns]
                v_out[m, :, self.factors] += cv[:self.columns]

        return v_out

    def reconstruct_b(self):
        """
        """
        v_out = np.zeros((self.rows, self.columns, self.factors))
        weights_cs = None
        for t in xrange(self.bases.shape[1]):
            if weights_cs is None:
                weights_cs = self.weights[:, 0, :]
            else:
                weights_cs = shift_column(weights_cs, 1)
            for r in xrange(self.factors):
                v_out[:, :, r] += np.dot(self.bases[:, t, r], weights_cs)
        return v_out

    def nmfdkl(self, pre, post):
        """
        """
        self.nmfdkl_iter(pre)
        self.post_process()
        self.nmfdkl_iter(post)
        return self.reconstruct()
