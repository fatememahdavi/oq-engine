# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2022, GEM Foundation
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
"""
Aftershock calculator
"""
import logging
import operator
import numpy
import pandas
from openquake.baselib import parallel, general
from openquake.calculators import base, preclassical

from openquake.hazardlib.aft.rupture_distances import (
    get_aft_rup_dists,
    prep_source_data,
)

from openquake.hazardlib.aft.aftershock_probabilities import (
    rupture_aftershock_rates_all_sources
)

U32 = numpy.uint32
F32 = numpy.float32


# NB: this is called after a preclassical calculation
def build_rates(srcs):
    """
    :param srcs: a list of split sources of the same source group
    """
    out = {'src_id': [], 'rup_id': [], 'delta': []}
    for src in srcs:
        for i, rup in enumerate(src.iter_ruptures()):
            out['src_id'].append(src.id)
            out['rup_id'].append(src.offset + i)
            # TODO: add the aftershock logic to compute the deltas
            # right now using a fake delta = 10% of the occurrence_rate
            try:
                delta = rup.occurrence_rate * .1
            except AttributeError:  # nonpoissonian rupture
                delta = numpy.nan
            out['delta'].append(delta)
    out['src_id'] = U32(out['src_id'])
    out['rup_id'] = U32(out['rup_id'])
    out['delta'] = F32(out['delta'])
    return pandas.DataFrame(out)


def get_aft_rup_rates(srcs, source_info=None,
    rup_dist_h5_file=None,
    dist_constant:float=4.0,
    max_block_ram: float=20.0,
    b_val: float = 1.25,
    alpha: float = 1.25,
    gr_max: float = 7.9,
    gr_min: float = 4.6,
    gr_bin_width: float = 0.1,
    c: float = 0.02,
    min_mainshock_mag: float = 5.0,):

    out = {'src_id': [], 'rup_id': [], 'delta': []}

    aft_rates = rupture_aftershock_rates_all_sources(srcs,
        source_info=source_info, gr_bin_width=gr_bin_width
        )

    for ind in aft_rates.index:
        out['src_id'].append(ind[0])
        out['rup_id'].append(ind[1])

    out['delta'] = aft_rates.values

    out['src_id'] = U32(out['src_id'])
    out['rup_id'] = U32(out['rup_id'])
    out['delta'] = F32(out['delta'])
    
    return pandas.DataFrame(out)


@base.calculators.add('aftershock')
class AftershockCalculator(preclassical.PreClassicalCalculator):
    """
    Aftershock calculator storing a dataset `delta_rates`
    """
    def post_execute(self, csm):
        logging.warning('Aftershock calculations are still experimental')
        self.datastore['_csm'] = csm
        sources = csm.get_sources()
        source_info = self.datastore["source_info"][:]

        df = get_aft_rup_rates(sources, source_info=source_info,
                                gr_bin_width=self.oqparam.width_of_mfd_bin)

        logging.info('Sorting rates')
        df = df.sort_values(['src_id', 'rup_id'])
        size = 0
        all_deltas = []
        num_ruptures = self.datastore['source_info']['num_ruptures']
        logging.info('Grouping deltas by %d src_id', len(num_ruptures))
        for src_id, grp in df.groupby('src_id'):
            # sanity check on the number of ruptures per source
            assert len(grp) == num_ruptures[src_id], (
                len(grp), num_ruptures[src_id])
            all_deltas.append(grp.delta.to_numpy())
            size += len(grp) * 4
        logging.info('Storing {} inside {}::/delta_rates'.format(
            general.humansize(size), self.datastore.filename))
        self.datastore.hdf5.save_vlen('delta_rates', all_deltas)
