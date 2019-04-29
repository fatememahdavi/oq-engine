# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2018 GEM Foundation
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
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`KothaEtAl2019`,
               :class:`KothaEtAl2019SERA`
"""
import os
import h5py
import numpy as np
from scipy.constants import g
from scipy.interpolate import interp1d
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA, from_string


BASE_PATH = os.path.join(os.path.dirname(__file__), "kotha_2019_tables")


class KothaEtAl2019(GMPE):
    """
    Implements the first complete version of the newly derived GMPE
    for Shallow Crustal regions using the Engineering Strong Motion Flatfile.

    (Working Title)
    Kotha, S. R., Weatherill, G., Bindi, D., Cotton F. (2019) A Revised
    Ground Motion Prediction Equation for Shallow Crustal Earthquakes in
    Europe and the Middle-East

    The GMPE is desiged for calibration of the stress parameter term
    (a multiple of the fault-to-fault variability, tau_f) an attenuation
    scaling term (c3) and a statistical uncertainty term (sigma_mu). The
    statistical uncertainty is a scalar factor dependent on period, magnitude
    and distance. These are read in from hdf5 upon instantiation and
    interpolated to the necessary values.

    In the core form of the GMPE no site term is included. This will be
    added in the subclasses.

    :param c3:
        User supplied table for the coefficient c3 controlling the anelastic
        attenuation as an instance of :class:
        `openquake.hazardlib.gsim.base.CoeffsTable`. If absent, the value is
        taken from the normal coefficients table.

    :param sigma_mu_epsilon:
        The number by which to multiply the epistemic uncertainty (sigma_mu)
        for the adjustment of the mean ground motion.
    """
    experimental = True

    #: Supported tectonic region type is 'active shallow crust'
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Set of :mod:`intensity measure types <openquake.hazardlib.imt>`
    #: this GSIM can calculate. A set should contain classes from module
    #: :mod:`openquake.hazardlib.imt`.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        PGV,
        SA
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD50

    #: Supported standard deviation types are inter-event, intra-event
    #: and total
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: Required site parameter is not set
    REQUIRES_SITES_PARAMETERS = set()

    #: Required rupture parameters are magnitude and hypocentral depth
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', 'hypo_depth'))

    #: Required distance measure is Rjb (eq. 1).
    REQUIRES_DISTANCES = set(('rjb', ))

    def __init__(self, sigma_mu_epsilon=0.0, c3=None):
        """
        Instantiate setting the sigma_mu_epsilon and c3 terms
        """
        super().__init__()
        if isinstance(c3, dict):
            # Inputing c3 as a dictionary sorted by the string representation
            # of the IMT
            c3in = {}
            for c3key in c3:
                c3in[from_string(c3key)] = {"c3": c3[c3key]}
            self.c3 = CoeffsTable(sa_damping=5, table=c3in)
        else:
            self.c3 = c3

        self.sigma_mu_epsilon = sigma_mu_epsilon
        print(self.c3, self.sigma_mu_epsilon)
        if self.sigma_mu_epsilon:
            # Connect to hdf5 and load tables into memory
            self.retreive_sigma_mu_data()
        else:
            # No adjustments, so skip this step
            self.mags = None
            self.dists = None
            self.s_a = None
            self.pga = None
            self.pgv = None
            self.periods = None

    def retreive_sigma_mu_data(self):
        """
        For the general form of the GMPE this retrieves the sigma mu
        values from the hdf5 file using the "general" model, i.e. sigma mu
        factors that are independent of the choice of region or depth
        """
        fle = h5py.File(os.path.join(BASE_PATH,
                                     "KothaEtAl2019_SigmaMu_Fixed.hdf5"), "r")
        self.mags = fle["M"][:]
        self.dists = fle["R"][:]
        self.periods = fle["T"][:]
        self.pga = fle["PGA"][:]
        self.pgv = fle["PGV"][:]
        self.s_a = fle["SA"][:]
        fle.close()

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extracting dictionary of coefficients specific to required
        # intensity measure type.
        C = self.COEFFS[imt]

        mean = (self.get_magnitude_scaling(C, rup.mag) +
                self.get_distance_term(C, rup, dists.rjb, imt) +
                self.get_site_amplification(C, sites))
        # GMPE originally in cm/s/s - convert to g
        if imt.name in "PGA SA":
            mean -= np.log(100.0 * g)
        stddevs = self.get_stddevs(C, dists.rjb.shape, stddev_types, sites)
        if self.sigma_mu_epsilon:
            # Apply the epistemic uncertainty factor (sigma_mu) multiplied by
            # the number of standard deviations
            sigma_mu = self.get_sigma_mu_adjustment(C, imt, rup, dists)
            # Cap sigma_mu at 0.5 ln units
            sigma_mu[sigma_mu > 0.5] = 0.5
            # Sigma mu should not be less than the standard deviation of the
            # fault-to-fault variability
            sigma_mu[sigma_mu < C["tau_fault"]] = C["tau_fault"]
            mean += (self.sigma_mu_epsilon * sigma_mu)
        return mean, stddevs

    def get_magnitude_scaling(self, C, mag):
        """
        Returns the magnitude scaling term
        """
        d_m = mag - self.CONSTANTS["Mh"]
        if mag < self.CONSTANTS["Mh"]:
            return C["e1"] + C["b1"] * d_m + C["b2"] * (d_m ** 2.0)
        else:
            return C["e1"] + C["b3"] * d_m

    def get_distance_term(self, C, rup, rjb, imt):
        """
        Returns the distance attenuation factor
        """
        h = self._get_h(C, rup.hypo_depth)
        rval = np.sqrt(rjb ** 2. + h ** 2.)
        c3 = self.get_distance_coefficients(C, imt)

        f_r = (C["c1"] + C["c2"] * (rup.mag - self.CONSTANTS["Mref"])) *\
            np.log(rval / self.CONSTANTS["Rref"]) +\
            c3 * (rval - self.CONSTANTS["Rref"])
        return f_r

    def _get_h(self, C, hypo_depth):
        """
        Returns the depth-specific coefficient
        """
        if hypo_depth <= 10.0:
            return C["h_D10"]
        elif hypo_depth > 20.0:
            return C["h_D20"]
        else:
            return C["h_10D20"]

    def get_distance_coefficients(self, C, imt):
        """
        Returns the c3 term
        """
        c3 = self.c3[imt]["c3"] if self.c3 else C["c3"]
        return c3

    def get_site_amplification(self, C, sites):
        """
        In base model no site amplification is used
        """
        return 0.0

    def get_sigma_mu_adjustment(self, C, imt, rup, dists):
        """
        Returns the sigma mu adjustment factor
        """
        if imt.name in "PGA PGV":
            # PGA and PGV are 2D arrays of dimension [nmags, ndists]
            sigma_mu = getattr(self, imt.name.lower())
            if rup.mag <= self.mags[0]:
                sigma_mu_m = sigma_mu[0, :]
            elif rup.mag >= self.mags[-1]:
                sigma_mu_m = sigma_mu[-1, :]
            else:
                intpl1 = interp1d(self.mags, sigma_mu, axis=0)
                sigma_mu_m = intpl1(rup.mag)
            # Linear interpolation with distance
            intpl2 = interp1d(self.dists, sigma_mu_m, bounds_error=False,
                              fill_value=(sigma_mu_m[0], sigma_mu_m[-1]))
            return intpl2(dists.rjb)
        # In the case of SA the array is of dimension [nmags, ndists, nperiods]
        # Get values for given magnitude
        if rup.mag <= self.mags[0]:
            sigma_mu_m = self.s_a[0, :, :]
        elif rup.mag >= self.mags[-1]:
            sigma_mu_m = self.s_a[-1, :, :]
        else:
            intpl1 = interp1d(self.mags, self.s_a, axis=0)
            sigma_mu_m = intpl1(rup.mag)
        # Get values for period - N.B. ln T, linear sigma mu interpolation
        if imt.period <= self.periods[0]:
            sigma_mu_t = sigma_mu_m[:, 0]
        elif imt.period >= self.periods[-1]:
            sigma_mu_t = sigma_mu_m[:, -1]
        else:
            intpl2 = interp1d(np.log(self.periods), sigma_mu_m, axis=1)
            sigma_mu_t = intpl2(np.log(imt.period))
        intpl3 = interp1d(self.dists, sigma_mu_t, bounds_error=False,
                          fill_value=(sigma_mu_t[0], sigma_mu_t[-1]))
        return intpl3(dists.rjb)

    def get_stddevs(self, C, stddev_shape, stddev_types, sites):
        """
        Returns the standard deviations
        """
        stddevs = []
        tau = C["tau_event"]
        phi = np.sqrt(C["phi0"] ** 2.0 + C["phis2s"] ** 2.)
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(np.sqrt(tau ** 2. + phi ** 2.) +
                               np.zeros(stddev_shape))
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(phi + np.zeros(stddev_shape))
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(tau + np.zeros(stddev_shape))
        return stddevs

    COEFFS = CoeffsTable(sa_damping=5, table="""\
      imt                 e1                 b1                  b2                 b3                  c1              h_D10             h_10D20               h_D20                 c2                  c3              tau_h             tau_c3             phis2s          tau_event          tau_fault               phi0            d0_obs              d1_obs        sigma_s_obs            d0_inf              d1_inf        sigma_s_inf
      pgv  1.892806328872520  1.997290013647580   0.086081063981519  0.630585733419827  -1.352927456741330   3.621177834846830  6.570188023667550  11.636086852028200  0.292226428741456  -0.003717368903319  3.350274740504120  0.001989175108629  0.597642034232805  0.372379668871244  0.311130314832717  0.483487360873615  3.59155588988794  -0.573104547503056  0.354252517655502  3.50154144252107  -0.549036348204835  0.471341743936194
      pga  4.417284059645490  1.381281569279050   0.004327491747107  0.378789009712617  -1.521958246846950   5.248835811278450  8.165564473597410  10.207830496829000  0.304956691600404  -0.006384808664692  1.230792745760920  0.003322073844615  0.669106254869499  0.381917344130535  0.370122543171531  0.510861957895391  2.79725327727529  -0.447626700422651  0.480117350959533  2.86975218395504  -0.450758427681675  0.565557973373843
    0.010  4.548444019710410  1.545413777369810   0.055777461826735  0.287004772074489  -1.518125079550980   5.196280657581540  8.103270060237860  10.539069934362600  0.298393283713967  -0.006458744312803  1.408668268220030  0.003249099924187  0.674745769793643  0.387148247525608  0.383687297188258  0.510104205329196  2.79727234832924  -0.447869656445290  0.481653781767443  2.87463675463566  -0.451742899490231  0.567276273220833
    0.025  4.442580815397090  1.346091126371840   0.000511785809480  0.422339574539637  -1.586970951009750   5.399704970497420  8.298529100299890  10.834714579137600  0.304615839626772  -0.005879118382059  1.393095036321120  0.003276969789544  0.671679004630200  0.383176945383864  0.366298823896753  0.512392066661141  2.70308959325038  -0.432005769213633  0.487280586089154  2.79414874529075  -0.438730483722306  0.569356945466973
    0.040  4.675805934816830  1.496300750855740   0.076016731610596  0.243275008695628  -1.728407882013360   5.942366776813880  8.587944926186650  12.624212787205000  0.319237677090743  -0.005120561634558  2.837493515680960  0.003112094247622  0.691848007494006  0.376564696325853  0.384505198625636  0.518461715489209  2.44415724128302  -0.390358911183250  0.527052093345632  2.73149628777198  -0.428908283902744  0.592284135200849
    0.050  4.656248766515670  1.254970613755230   0.003614618720890  0.316737720903129  -1.794320830857420   6.503489728482530  9.136811109794560  12.569039314120400  0.332108140841410  -0.005264500214593  1.827537822442560  0.003353817406823  0.704483726226435  0.374527074391120  0.403501921390989  0.524129983454523  2.39487265788737  -0.382896939598838  0.545580900582058  2.69008768532356  -0.422278148438213  0.609760771097008
    0.070  4.922575670299870  1.263395221116320   0.014010962989755  0.319746235272060  -1.826812962184560   7.416957296597530  10.05184974781670  12.829896514151700  0.342173772868787  -0.005657375107420  1.673134498670350  0.003756011346075  0.730696638333531  0.377132478863452  0.433435245514180  0.527258463232455  2.36043303535580  -0.380433120317511  0.558959254429671  2.69088202174411  -0.423030530282438  0.637249070963625
    0.100  5.273573951455780  1.455230368535440   0.072738302170768  0.165303361562237  -1.718933300761690   7.767511226547350  10.32601427372340  13.151816895596300  0.309586419036125  -0.006883472019433  2.271207407275640  0.004231945903474  0.753507140006929  0.382040067526270  0.433812760529140  0.534721815465482  2.28643533414677  -0.368234009748755  0.591287812067563  2.65702415193391  -0.418380962130904  0.659444131894117
    0.150  5.365005467478860  1.455572275342220   0.040440987249440  0.243034969330473  -1.430653323847180   5.810656994201750  9.082810077383200  11.337820953209100  0.247751883884461  -0.008379507529490  2.419548710426230  0.004285093038848  0.742264291850245  0.399169524678397  0.410545124538557  0.536676157950607  2.58767038578537  -0.415628772851415  0.578608595614738  2.66768751935281  -0.419695530569169  0.646231316632271
    0.200  5.340843250207400  1.421317129241390  -0.008574793393420  0.404597294230414  -1.300571451166200   4.675853256994330  8.707712056336570  11.331373707888200  0.200900820815253  -0.008258443814965  2.474519037517560  0.004065520686124  0.725320656640016  0.410970785906127  0.384111554440299  0.532188765446470  3.06257295397241  -0.490955029234259  0.540015369367897  2.84959984752439  -0.448452245334260  0.621110826966698
    0.250  5.379165191094910  1.613172422465060   0.030732401474054  0.282107490214437  -1.224456747634730   3.931649778228820  8.283466589346060  10.699022985920500  0.174457473156904  -0.008168225448742  2.934369251816590  0.003694223510441  0.702963159310018  0.412506846837313  0.373580297647849  0.526995694991015  3.13458492347708  -0.504668552535490  0.501353212589825  2.95547915669062  -0.465462087344715  0.594501521043147
    0.300  5.290651826542550  1.677045706210000   0.028582516747067  0.300455509720834  -1.179087781402620   3.571988098279060  7.530686758925950  10.601807863073700  0.160489044877199  -0.007633705781879  2.944948091989140  0.003523170130238  0.691160893394519  0.409290395416361  0.361190381465491  0.521904513113003  3.24558518828961  -0.522520462500027  0.477618221616811  3.11399537945149  -0.490182911905075  0.576990047530632
    0.350  5.143420445805400  1.676699374038290   0.011926131038970  0.413771449361342  -1.151384769209340   3.478098007259010  7.264029052679490   9.638426432589590  0.149835683999565  -0.006957073673618  2.726977604394550  0.003380818237915  0.677871583212350  0.404722333499924  0.349932304751709  0.517473487991425  3.44838796236805  -0.552467586074445  0.454490665715818  3.32660997337116  -0.523265793495923  0.558934733460938
    0.400  5.007766105042600  1.654070453039360  -0.014767594472343  0.502752266070053  -1.120171296841220   3.316580856168640  7.014295416608370   8.942947523048470  0.142932983815086  -0.006621400220309  2.297576377223100  0.003224354754275  0.676342491986797  0.402216454290276  0.349594773894112  0.510214230044972  3.73614899367811  -0.597200285087204  0.446340645075208  3.49321148509485  -0.549187545850890  0.553365815976012
    0.450  4.993309405388980  1.787861125523350   0.013327295813481  0.382735050831296  -1.116490216706990   3.303587351513180  6.914732372738140   8.748850430484320  0.138194933304793  -0.006096405733691  2.430095568240010  0.003154801713776  0.678495425690818  0.399307399487995  0.343276544713056  0.503982756931627  3.94978634376916  -0.630839152367926  0.444619048959004  3.66166095243848  -0.575601401138774  0.554477875004125
    0.500  5.025808775293170  2.010879653962750   0.072935798481947  0.253212667760166  -1.125872461848280   3.313175921989680  6.705696316546080   8.781889076061700  0.133115624510051  -0.005479207002881  1.575567280567830  0.003026941234605  0.679453093268171  0.402086391093198  0.324698722709901  0.496744390517092  3.97043265095054  -0.633280800189855  0.442401729050217  3.80437373293752  -0.597770227095264  0.551767933300466
    0.600  4.575054661557320  1.627559080374890  -0.081644115008946  0.636839453474050  -1.122896467461930   3.250388459820300  6.755141447727090  10.578310335521400  0.123781303419217  -0.004644786062988  1.762812620619740  0.002686696245752  0.680693821092247  0.398582471430753  0.326299644912384  0.485887400763896  4.11553965350644  -0.655519404806555  0.431767437492638  4.01195042750724  -0.630227822234957  0.549925656586413
    0.700  4.552997806717560  1.855279432723890  -0.036656519663037  0.320315409452011  -1.124261988244940   3.164458715155760  6.795706851017660  14.428096427586800  0.116709781985408  -0.004025744863412  2.897309755112500  0.002564264366739  0.681285118343840  0.405350717442049  0.311798101479078  0.477975448246383  4.40373663774109  -0.700058288174968  0.419186105021081  4.12174760391623  -0.646775405781893  0.548180262000477
    0.750  4.490071505295000  1.954274125293150  -0.003641660433376  0.456746317851063  -1.100945904099820   2.993445369118350  6.605231810094700  14.816521542859300  0.118164255034819  -0.004061356198934  4.344844406860950  0.002396351691290  0.675374259014311  0.399886897132088  0.308310368674288  0.474831748313580  4.41370691686027  -0.700929994258438  0.419235647299442  4.11858834742546  -0.646311647282336  0.544827942346524
    0.800  4.346340135789670  1.903954443525600  -0.027210137253621  0.571405031282990  -1.091284189625060   3.011537576407330  6.383813030413980   8.654923981735060  0.117888128784975  -0.003879315832285  1.064916407668700  0.002262050918392  0.669964265091023  0.405011617726239  0.297779075583881  0.470675816183951  4.40023807557548  -0.698286286632014  0.420206179657632  4.15262278426163  -0.651443005295049  0.543149430752588
    0.900  4.172884941038480  1.870903240776090  -0.059326413467110  0.641520889426695  -1.083165017606630   2.914691523196970  6.203853239158910  13.416728990608000  0.117605536642874  -0.003666723557583  2.969877409593050  0.002162505133793  0.682312548536914  0.401371185079930  0.305551357825079  0.461879366570655  4.54974665381206  -0.721274094012117  0.433447566803411  4.33437738669094  -0.679801540800839  0.549339600089621
    1.000  4.273364795550890  2.284923556328090   0.065866868784990  0.390273018724367  -1.066847039367690  -2.770998201229750  5.860078215476920  13.460444813265200  0.120495994214884  -0.003428313609870  3.881503254169090  0.001961940063713  0.679308403598395  0.406075656060488  0.318741184949152  0.455552800346687  4.63320008058818  -0.733070104878305  0.435831282783962  4.39109255455187  -0.688465075188292  0.545809019593528
    1.200  3.937942766920120  2.170455024448590   0.001174567307882  0.611504555404356  -1.066799402049050  -2.802467021249190  5.853186666416320  14.231560539244100  0.131385480451062  -0.002911204978252  5.399650832898010  0.001827548291076  0.672245643532270  0.402070918575962  0.309254551116353  0.444878350249610  4.66293055224909  -0.737895073569996  0.434028645187010  4.39765274799130  -0.689425377174240  0.543229115708139
    1.400  3.772961844544640  2.291407004764750   0.019686183706859  0.639178281399829  -1.070485349817400   2.964269815258380  5.623451439084210   7.746278822950370  0.130566705424876  -0.002529697033308  0.849207406682328  0.001567754308089  0.662596356519541  0.407601922856431  0.292848034404669  0.436849421688952  4.50690099561210  -0.713882430127437  0.440012962336114  4.35234175521656  -0.682441565982256  0.540827372209902
    1.600  3.514467366093290  2.294895809430900   0.010752267351047  0.713970090920851  -1.091228676769790   2.883405145840870  5.644671551941680  14.525659107086500  0.144029228752413  -0.002047277539427  5.089802482507260  0.001586628710920  0.664551801117703  0.410886432394217  0.300675754668513  0.431989239017968  4.42698771453499  -0.701598682161155  0.444141743600401  4.35714361415801  -0.682557084627090  0.539116843859117
    1.800  3.558886245279490  2.666010152380630   0.115262687271212  0.497582334875118  -1.100660898745970   2.949559265826990  5.480402245803100  10.216862145575300  0.146617369543987  -0.001554451068759  1.296530314251750  0.001390149466717  0.665502891686339  0.389699175362844  0.287717357430583  0.428737629779977  4.38722282924947  -0.696767820891729  0.448128375603855  4.34980568924963  -0.681312244335941  0.534908929898277
    2.000  3.183532894947410  2.388446315455500   0.018352451718499  0.584383533482738  -1.108358161998800  -2.911819519994510  5.278887669847590  15.668917064866600  0.150743827092867  -0.001580015449440  7.687533056277530  0.001354445228794  0.656519021204742  0.390879045937366  0.261303789411214  0.427120153573043  4.26618002948656  -0.677840863946992  0.445268405111334  4.36909584809318  -0.684254142482054  0.529187650914867
    2.500  2.836907451195650  2.536546088155800   0.062398760604848  0.596079144473462  -1.105603366221790  -2.823586071191350  4.750273265887600  15.487892191774300  0.175239687999456  -0.001631228333605  5.461836301977410  0.001201195688135  0.646055736219785  0.379738530038140  0.223229075011747  0.425817651524798  4.22371310321513  -0.673186562665730  0.422435774484840  4.33349112967483  -0.678761193852883  0.519689843919031
    3.000  2.517737034750380  2.546050164341940   0.055154674642950  0.809850456474428  -1.080067562920250   2.791528486447530  4.198987339784080  16.199213420987600  0.184128987340750  -0.001922721370983  5.495712825361990  0.001266687383427  0.644137229457818  0.381565976193432  0.242099243686895  0.422252639239821  4.09589755151420  -0.655102179588055  0.415188078320107  4.28612060333917  -0.671097390391303  0.516644494740873
    3.500  2.119154537449590  2.396069299049670   0.008539272672874  1.164554654673330  -1.078957829171880   2.799516490028090  3.809647060408550  14.361952992707000  0.192610794489156  -0.002126518816230  2.988237308109300  0.001270039982803  0.619042168874467  0.366597795968384  0.248890074124623  0.421148406228811  3.87130345045571  -0.620935593452115  0.393657395953868  4.05518974023777  -0.635183933808111  0.501852480973462
    4.000  1.847621951421810  2.360501647907130  -0.012290291175221  1.203181384723020  -1.085147369608560   2.850640140168700  3.753076646146860   7.572223679128170  0.212495563316293  -0.002160985939694  0.972813090643118  0.001160548266474  0.611445419916175  0.368313505931245  0.253499095223261  0.423003337400490  3.66750003456413  -0.588258931615559  0.377650627209691  3.88014398605592  -0.607581511885431  0.492187585551506
    4.500  2.004721331888600  2.749583105839750   0.124920421668848  1.109962973543350  -1.106053570513110   3.052905831797360  4.057744203916230   4.310346595023010  0.222730247969709  -0.001860522670072  0.709961258389252  0.000000000000000  0.611455447150517  0.357487272025988  0.241635677822731  0.408704987873161  3.50840501651649  -0.565538752485990  0.409557470904523  4.00798424133467  -0.626644011126345  0.489116498441993
    5.000  1.766244279359640  2.775644704909250   0.136811181908039  1.277024063081970  -1.134940922975580   3.156427441541180  4.182103070858070   4.099969082316580  0.247128194747863  -0.001550196672696  0.625986200151889  0.000000000000000  0.598171990718161  0.357769751705229  0.250782161006834  0.409277420621721  3.27278257182859  -0.529012351103819  0.397002875689376  3.84718134870989  -0.601713782510972  0.476960884877633
    6.000  1.495644634092600  2.906956649834780   0.206605866198126  1.436150033009390  -1.155226946853440   2.732111686024190  5.157196005146210   5.134139700462980  0.261450980486218  -0.001439766119884  1.343568797412350  0.000000000000000  0.577497461758988  0.364322510724063  0.224152579183778  0.401597199934827  3.01673206668580  -0.489064156495419  0.379679274738408  3.58229829260865  -0.561049974091566  0.459592859585885
    7.000  1.182821190754860  2.917667033954530   0.223647052589440  1.645537964856860  -1.228574871054330   2.970825808343180  5.783138212668300   5.347061845422780  0.289575079930824  -0.000867891160389  1.439390706721320  0.000000000000000  0.563819357916120  0.368571233598105  0.229328741533368  0.401052676368561  3.09654599365856  -0.503275832203786  0.363962303516824  3.44766788599524  -0.540181500989622  0.447668743764740
    8.000  0.877936104327193  2.908933205097090   0.234947578468289  1.691937861016660  -1.271209905469500   3.052069542435400  6.188468047371990   6.198930745809970  0.312385151596051  -0.000600200472803  1.671791342658880  0.000892670164182  0.556359572194014  0.365797914105721  0.204945101304949  0.401323166589614  3.08002142877393  -0.501377187109764  0.359647250033555  3.38667681338371  -0.530743607500501  0.440968433335620
    """)

    CONSTANTS = {"Mref": 4.5, "Rref": 30., "Mh": 6.2}


class KothaEtAl2019SERA(KothaEtAl2019):
    """
    Implementation of the Kotha et al. (2019) GMPE with the site
    amplification components included. This form of the GMPE defines the
    site in terms of a measured or inferred Vs30, with the total
    aleatory variability adjusted accordingly.
    """

    #: Required site parameter is not set
    REQUIRES_SITES_PARAMETERS = set(("vs30", "vs30measured"))

    def get_site_amplification(self, C, sites):
        """
        Returns the linear site amplification term depending on whether the
        Vs30 is observed of inferred
        """
        ampl = np.zeros(sites.vs30.shape)
        # For observed vs30 sites
        ampl[sites.vs30measured] = (C["d0_obs"] + C["d1_obs"] *
                                    np.log(sites.vs30[sites.vs30measured]))
        # For inferred Vs30 sites
        idx = np.logical_not(sites.vs30measured)
        ampl[idx] = (C["d0_inf"] + C["d1_inf"] * np.log(sites.vs30[idx]))
        return ampl

    def get_stddevs(self, C, stddev_shape, stddev_types, sites):
        """
        Returns the standard deviations, with different site standard
        deviation for inferred vs. observed vs30 sites.
        """
        stddevs = []
        tau = C["tau_event"]
        sigma_s = np.zeros(sites.vs30measured.shape, dtype=float)
        sigma_s[sites.vs30measured] += C["sigma_s_obs"]
        sigma_s[np.logical_not(sites.vs30measured)] += C["sigma_s_inf"]
        phi = np.sqrt(C["phi0"] ** 2.0 + sigma_s ** 2.)
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(np.sqrt(tau ** 2. + phi ** 2.) +
                               np.zeros(stddev_shape))
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(phi + np.zeros(stddev_shape))
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(tau + np.zeros(stddev_shape))
        return stddevs
