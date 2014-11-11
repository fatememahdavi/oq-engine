# The Hazard Library
# Copyright (C) 2014, GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import (
    CampbellBozorgnia2014,
    CampbellBozorgnia2014HighQ,
    CampbellBozorgnia2014LowQ,
    CampbellBozorgnia2014JapanSite,
    CampbellBozorgnia2014HighQJapanSite,
    CampbellBozorgnia2014LowQJapanSite)

from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase


class CampbellBozorgnia2014TestCase(BaseGSIMTestCase):
    GSIM_CLASS = CampbellBozorgnia2014
    MEAN_FILE = 'CB14/CB2014_MEAN.csv'
    STD_INTRA_FILE = 'CB14/CB2014_STD_INTRA.csv'
    STD_INTER_FILE = 'CB14/CB2014_STD_INTER.csv'
    STD_TOTAL_FILE = 'CB14/CB2014_STD_TOTAL.csv'

    def test_mean(self):
        self.check(self.MEAN_FILE,
                   max_discrep_percentage=0.1)

    def test_std_intra(self):
        self.check(self.STD_INTRA_FILE,
                   max_discrep_percentage=0.1)

    def test_std_inter(self):
        self.check(self.STD_INTER_FILE,
                   max_discrep_percentage=0.1)

    def test_std_total(self):
        self.check(self.STD_TOTAL_FILE,
                   max_discrep_percentage=0.1)


class CampbellBozorgnia2014HighQTestCase(BaseGSIMTestCase):
    GSIM_CLASS = CampbellBozorgnia2014HighQ
    MEAN_FILE = 'CB14/CB2014_HIGHQ_MEAN.csv'
    STD_INTRA_FILE = 'CB14/CB2014_HIGHQ_STD_INTRA.csv'
    STD_INTER_FILE = 'CB14/CB2014_HIGHQ_STD_INTER.csv'
    STD_TOTAL_FILE = 'CB14/CB2014_HIGHQ_STD_TOTAL.csv'


class CampbellBozorgnia2014LowQTestCase(BaseGSIMTestCase):
    GSIM_CLASS = CampbellBozorgnia2014LowQ
    MEAN_FILE = 'CB14/CB2014_LOWQ_MEAN.csv'
    STD_INTRA_FILE = 'CB14/CB2014_LOWQ_STD_INTRA.csv'
    STD_INTER_FILE = 'CB14/CB2014_LOWQ_STD_INTER.csv'
    STD_TOTAL_FILE = 'CB14/CB2014_LOWQ_STD_TOTAL.csv'


class CampbellBozorgnia2014JapanSiteTestCase(BaseGSIMTestCase):
    GSIM_CLASS = CampbellBozorgnia2014JapanSite
    MEAN_FILE = 'CB14/CB2014_JAPAN_MEAN.csv'
    STD_INTRA_FILE = 'CB14/CB2014_JAPAN_STD_INTRA.csv'
    STD_INTER_FILE = 'CB14/CB2014_JAPAN_STD_INTER.csv'
    STD_TOTAL_FILE = 'CB14/CB2014_JAPAN_STD_TOTAL.csv'


class CampbellBozorgnia2014HighQJapanSiteTestCase(BaseGSIMTestCase):
    GSIM_CLASS = CampbellBozorgnia2014HighQJapanSite
    MEAN_FILE = 'CB14/CB2014_HIGHQ_JAPAN_MEAN.csv'
    STD_INTRA_FILE = 'CB14/CB2014_HIGHQ_JAPAN_STD_INTRA.csv'
    STD_INTER_FILE = 'CB14/CB2014_HIGHQ_JAPAN_STD_INTER.csv'
    STD_TOTAL_FILE = 'CB14/CB2014_HIGHQ_JAPAN_STD_TOTAL.csv'


class CampbellBozorgnia2014LowQJapanSiteTestCase(BaseGSIMTestCase):
    GSIM_CLASS = CampbellBozorgnia2014LowQJapanSite
    MEAN_FILE = 'CB14/CB2014_LOWQ_JAPAN_MEAN.csv'
    STD_INTRA_FILE = 'CB14/CB2014_LOWQ_JAPAN_STD_INTRA.csv'
    STD_INTER_FILE = 'CB14/CB2014_LOWQ_JAPAN_STD_INTER.csv'
    STD_TOTAL_FILE = 'CB14/CB2014_LOWQ_JAPAN_STD_TOTAL.csv'
