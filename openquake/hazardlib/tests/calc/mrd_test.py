# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2023, GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from openquake.baselib.performance import Monitor
from openquake.hazardlib.calc.mrd import (
    update_mrd, get_uneven_bins_edges, calc_mean_rate_dist)
from openquake.hazardlib.contexts import read_cmakers
from openquake.commonlib import datastore
from openquake.hazardlib.cross_correlation import BakerJayaram2008

PLOT = False
CWD = os.path.dirname(__file__)


class MRD01TestCase(unittest.TestCase):
    """ Computes the mean rate density using a simple PSHA input model """

    def setUp(self):

        # Settings
        fname = os.path.join(CWD, 'data', 'mrd', 'calc_934.hdf5')
        self.imts = ['SA(0.2)', 'SA(1.0)']  # subset of the parent IMTs

        # Load datastore
        self.dstore = datastore.read(fname)
        self.oqp = self.dstore['oqparam']

        # Read the context maker and set the IMTLS
        [cmaker] = read_cmakers(self.dstore)
        self.cmaker = cmaker.restrict(self.imts)

        # Read contexts
        [self.ctx] = self.cmaker.read_ctxs(self.dstore)

        # Set the cross correlation model
        self.crosscorr = BakerJayaram2008()

    def test(self):
        # Compute the MRD
        imls1 = self.oqp.imtls[self.imts[0]]
        imls2 = self.oqp.imtls[self.imts[1]]
        len1 = len(imls1) - 1
        len2 = len(imls2) - 1
        mrd = np.zeros((len1, len2, len(self.cmaker.gsims)))
        update_mrd(self.ctx, self.cmaker, self.crosscorr, mrd)

        # Loading Hazard Curves.
        # The poes array is 4D: |sites| x |stats| x |IMTs| x |IMLs|
        poes = self.dstore['hcurves-stats'][:]
        afe = - np.log(1-poes)
        afo = afe[:, :, :, :-1] - afe[:, :, :, 1:]

        imts = list(self.oqp.imtls)
        idx1 = imts.index(self.imts[0])
        idx2 = imts.index(self.imts[1])

        afo1 = afo[0, 0, idx1, :]
        afo2 = afo[0, 0, idx2, :]

        c1 = imls1[:-1] + np.diff(imls1) / 2
        c2 = imls2[:-1] + np.diff(imls2) / 2

        # Compute marginal
        marg1 = mrd.sum(axis=0)[:, 0]  # shape (44, 1) => 44
        marg2 = mrd.sum(axis=1)[:, 0]  # shape (44, 1) => 44

        # Test
        np.testing.assert_almost_equal(marg1, afo1, decimal=5)
        np.testing.assert_almost_equal(marg2, afo2, decimal=5)

        if PLOT:
            plt.title('Direct method test')
            plt.plot(c1, afo1, label=f'HC {self.imts[0]}')
            plt.plot(c1, marg1, 'o', mfc='None')
            plt.plot(c2, afo2, label=f'HC {self.imts[1]}')
            plt.plot(c2, marg2, 'o', mfc='None', )
            plt.xlabel('Spectral Acceleration, S$_a$ [g]')
            plt.ylabel('Annual Rate of Occurrence')
            plt.legend()
            plt.grid(which='minor', ls=':', color='lightgrey')
            plt.grid(which='major', ls='--', color='grey')
            plt.show()

        ## test_indirect

        # Bin edges
        lefts = [-3, -2, 1, 2]
        numb = [80, 80, 10]
        be_mea = get_uneven_bins_edges(lefts, numb)
        be_sig = np.arange(0.50, 0.70, 0.01)
        imt1, imt2 = self.imts

        # Compute the MRD
        mon = Monitor('multivariate')
        mrdi = calc_mean_rate_dist(self.ctx, 1, self.cmaker, self.crosscorr,
                                   imt1, imt2, be_mea, be_sig, mon)
        print(mon)

        # Compute marginal
        marg1 = mrdi.sum(axis=0)[:, 0, 0]  # shape (44, 1, 1) => 44
        marg2 = mrdi.sum(axis=1)[:, 0, 0]  # shape (44, 1, 1) => 44

        # Test
        np.testing.assert_almost_equal(marg1, afo1, decimal=5)
        np.testing.assert_almost_equal(marg2, afo2, decimal=5)

        if PLOT:
            plt.title('Indirect method test')
            plt.plot(c1, afo1, label=f'HC {self.imts[0]}')
            plt.plot(c1, marg1, 'o', mfc='None')
            plt.plot(c2, afo2, label=f'HC {self.imts[1]}')
            plt.plot(c2, marg2, 'o', mfc='None', )
            plt.xlabel('Spectral Acceleration, S$_a$ [g]')
            plt.ylabel('Annual Rate of Occurrence')
            plt.legend()
            plt.grid(which='minor', ls=':', color='lightgrey')
            plt.grid(which='major', ls='--', color='grey')
            plt.show()

        ## test_compare

        # Bin edges
        be_mea = get_uneven_bins_edges([-3, -2, 1, 2], [80, 80, 10])
        be_sig = np.arange(0.50, 0.70, 0.01)
        np.testing.assert_almost_equal(mrdi[:, :, 0], mrd)

        if PLOT:
            fig, axs = plt.subplots(1, 1)
            fig.set_size_inches(9, 6)
            plt1 = plt.contourf(np.log(c1), np.log(c2), mrd[:, :, 0, 0])
            _ = plt.contour(np.log(c1), np.log(c2), mrdi[:, :, 0, 0],
                            colors='orange', linestyles='dashed')
            _ = plt.colorbar(mappable=plt1)
            _ = plt.title('MRD')
            axs.set_xlabel(f'{self.imts[0]}')
            axs.set_ylabel(f'{self.imts[1]}')
            plt.show()
