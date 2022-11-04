import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pytest

from openquake.hazardlib.geo import Polygon, Point
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.source.area import AreaSource
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.geo.nodalplane import NodalPlane

from rupture_distances import (
    # RupDistType,
    # calc_min_source_dist,
    get_close_source_pairs,
    # calc_pairwise_distances,
    # min_reduce,
    # stack_sequences,
    # split_rows,
    # get_min_rup_dists,
    # check_dists_by_mag,
    # filter_dists_by_mag,
    # get_rup_dist_pairs,
    # process_source_pair,
    calc_rupture_adjacence_dict_all_sources,
    prep_source_data,
)

from aftershock_probabilities import (
    get_aftershock_grmfd,
    num_aftershocks,
    get_a,
    get_source_counts,
    get_aftershock_rup_rates,
    get_rup,
    RupDist2,
    make_source_dist_df,
    fetch_rup_from_source_dist_groups,
    rupture_aftershock_rates_per_source,
)

area_source_1 = AreaSource(
    source_id="s1",
    name="s1",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0),
            Point(0.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)

area_source_2 = AreaSource(
    source_id="s2",
    name="s2",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(2.0, 0.0, 0.0),
            Point(2.0, -1.0, 0.0),
            Point(3.0, -1.0, 0.0),
            Point(3.0, 0.0, 0),
            Point(2.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)

area_source_3 = AreaSource(
    source_id="s3",
    name="s3",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(4.0, 0.0, 0.0),
            Point(4.0, 1.0, 0.0),
            Point(5.0, 1.0, 0.0),
            Point(5.0, 0.0, 0),
            Point(4.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)


def test_num_aftershocks_1():

    Mmain = np.linspace(4.5, 8.5)
    c = 0.015
    alpha = 1.0

    num_afts = np.array(
        [
            474,
            572,
            690,
            833,
            1006,
            1214,
            1465,
            1768,
            2133,
            2575,
            3107,
            3750,
            4525,
            5461,
            6590,
            7953,
            9598,
            11583,
            13979,
            16869,
            20358,
            24568,
            29648,
            35780,
            43179,
            52108,
            62884,
            75887,
            91581,
            110519,
            133373,
            160954,
            194238,
            234406,
            282879,
            341376,
            411971,
            497163,
            599973,
            724043,
            873770,
            1054459,
            1272514,
            1535660,
            1853224,
            2236457,
            2698940,
            3257061,
            3930597,
            4743416,
        ]
    )

    np.testing.assert_equal(num_aftershocks(Mmain, c=c, alpha=alpha), num_afts)


def test_get_a():
    Mmain = np.linspace(4.5, 8.5)
    c = 0.015
    alpha = 1.0
    a_vals = np.array(
        [
            2.67577834,
            2.75739603,
            2.83884909,
            2.920645,
            3.00259798,
            3.08421869,
            3.16583762,
            3.24748226,
            3.32899086,
            3.41077723,
            3.49234125,
            3.57403127,
            3.65561858,
            3.73727218,
            3.81888541,
            3.90053098,
            3.98218075,
            4.06382106,
            4.1454761,
            4.22708934,
            4.30873511,
            4.3903698,
            4.4719954,
            4.55364034,
            4.63527258,
            4.7169044,
            4.79854016,
            4.88016738,
            4.96180538,
            5.04343695,
            5.12506792,
            5.20670177,
            5.2883342,
            5.36996872,
            5.45160071,
            5.53323299,
            5.61486665,
            5.6964988,
            5.77813171,
            5.85976436,
            5.94139713,
            6.0230297,
            6.10466257,
            6.18629507,
            6.26792792,
            6.34956055,
            6.43119323,
            6.51282589,
            6.59445852,
            6.67609121,
        ]
    )

    np.testing.assert_equal(get_a(Mmain, c=c, alpha=alpha))


def test_get_aftershock_grmfd():
    pass


def test_get_aftershock_rup_rates_1():
    # set up
    pass


def get_aftershock_rup_adjustments(b_val=1.0, alpha=1.0, gr_max=7.9, c=0.35):

    sources = [area_source_1, area_source_2, area_source_3]
    rup_df, source_groups = prep_source_data(sources)

    source_pairs = get_close_source_pairs(sources)

    rup_dists = calc_rupture_adjacence_dict_all_sources(
        source_pairs, rup_df, source_groups
    )

    source_counts, source_cum_counts, source_count_starts = get_source_counts(
        sources
    )

    rup_adjustments = []

    r_on = 1
    for ns, source in enumerate(sources):
        rup_adjustments.extend(
            rupture_aftershock_rates_per_source(
                source.source_id,
                rup_dists,
                source_count_starts=source_count_starts,
                rup_df=rup_df,
                source_groups=source_groups,
                r_on=r_on,
                ns=ns,
                min_mag=5.0,
                c=c,
                b_val=b_val,
                alpha=alpha,
                gr_max=gr_max,
            )
        )
        r_on = source_cum_counts[ns] + 1

    rr = [r for r in rup_adjustments if len(r) != 0]

    rup_adj_df = pd.concat([pd.DataFrame(r) for r in rr], axis=1).fillna(0.0)

    rup_adjustments = rup_adj_df.sum(axis=1)

    return rup_adjustments


def mag_to_mo(mag: float, c: float = 9.05):
    """
    Scalar moment [in Nm] from moment magnitude
    :return:
        The computed scalar seismic moment
    """
    return 10 ** (1.5 * mag + c)


def plot_mfds():
    sources = [area_source_1, area_source_2, area_source_3]
    rup_df, source_groups = prep_source_data(sources)

    rup_adjustments_alpha_b_1_reg = get_aftershock_rup_adjustments()
    rup_adjustments_alpha_1_5_reg = get_aftershock_rup_adjustments(alpha=1.5)
    rup_adjustments_alpha_b_1_5_reg = get_aftershock_rup_adjustments(
        alpha=1.25, b_val=1.25, c=0.15
    )
    rup_adjustments_alpha_Mrup = get_aftershock_rup_adjustments(gr_max=None)

    rup_df["rates"] = [rup.occurrence_rate for rup in rup_df.rupture]

    aft_rup_rates = pd.Series(index=rup_df.index, data=np.zeros(len(rup_df)))

    aft_rup_rates_a_b_1_reg = aft_rup_rates.add(
        rup_adjustments_alpha_b_1_reg, fill_value=0.0
    )
    # aft_rup_rates_alpha_1_5_reg = aft_rup_rates.add(rup_adjustments_alpha_1_5_reg, fill_value=0.0)
    aft_rup_rates_a_1_5_reg = aft_rup_rates.add(
        rup_adjustments_alpha_1_5_reg, fill_value=0.0
    )
    aft_rup_rates_a_b_1_5_reg = aft_rup_rates.add(
        rup_adjustments_alpha_b_1_5_reg, fill_value=0.0
    )
    aft_rup_rates_a_b_1_Mrup = aft_rup_rates.add(
        rup_adjustments_alpha_Mrup, fill_value=0.0
    )

    # rates_w_aftershocks = rup_df.rates + aft_rup_rates
    rates_w_aftershocks = rup_df.rates + aft_rup_rates_a_b_1_reg
    rates_w_aftershocks_1_5 = rup_df.rates + aft_rup_rates_a_1_5_reg
    rates_w_aftershocks_a_b_1_5 = rup_df.rates + aft_rup_rates_a_b_1_5_reg
    rates_w_aftershocks_a_b_1_Mrup = rup_df.rates + aft_rup_rates_a_b_1_Mrup

    mag_arg_sort = np.argsort(rup_df["mag"])[::-1]

    mag_sort = rup_df["mag"].values[mag_arg_sort]

    cum_rates = np.cumsum(rup_df.rates.values[mag_arg_sort])
    cum_aft_rates = np.cumsum(rates_w_aftershocks.values[mag_arg_sort])
    cum_aft_rates_a_1_5 = np.cumsum(
        rates_w_aftershocks_1_5.values[mag_arg_sort]
    )
    cum_aft_rates_a_b_1_5 = np.cumsum(
        rates_w_aftershocks_a_b_1_5.values[mag_arg_sort]
    )
    cum_aft_rates_a_b_1_Mrup = np.cumsum(
        rates_w_aftershocks_a_b_1_Mrup.values[mag_arg_sort]
    )

    plt.figure()

    plt.semilogy(mag_sort, cum_rates, label="mainshocks")
    plt.semilogy(mag_sort, cum_aft_rates, label="alpha, b = 1.0", linestyle="-")
    # plt.semilogy(
    #     mag_sort, cum_aft_rates_a_1_5, label="alpha= 1.5", linestyle="-."
    # )
    plt.semilogy(
        mag_sort, cum_aft_rates_a_b_1_5, label="alpha, b = 1.5", linestyle=":"
    )
    plt.semilogy(
        mag_sort,
        cum_aft_rates_a_b_1_Mrup,
        label="gr_max = Mmain",
        linestyle="-",
    )
    plt.legend()
    plt.title("Mainshock and aftershock MFDs")
    plt.show()

    return


def look_at_aftershock_rup_rates():
    sources = [area_source_1, area_source_2, area_source_3]
    rup_df, source_groups = prep_source_data(sources)
    source_pairs = get_close_source_pairs(sources)

    rup_dists = calc_rupture_adjacence_dict_all_sources(
        source_pairs, rup_df, source_groups
    )

    source_counts, source_cum_counts, source_count_starts = get_source_counts(
        sources
    )


plot_mfds()
