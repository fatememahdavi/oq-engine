# coding: utf-8
# The Hazard Library
# Copyright (C) 2012 GEM Foundation
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
from openquake.hazardlib.gsim.base import CoeffsTable

COEFFS_FS_ROCK_SWISS05 = CoeffsTable(sa_damping=5, table="""\
       IMT    k_adj           a1              a2              b1              b2              Rm             phi_11     phi_21  C2      Mc1     Mc2     Rc11        Rc21        mean_phi_ss
       pga    0.893272000     1.642085E+00    1.833001E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.58000    0.47000 0.35000 5.00000 7.00000 16.00000    36.00000    0.46
       0.05   0.960326204     1.590620E+00    1.617332E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.55204    0.44903 0.40592 5.00000 7.00000 16.00000    36.00000    0.4530103
       0.10   0.883646750     1.478437E+00    1.608714E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.54000    0.44000 0.43000 5.00000 7.00000 16.00000    36.00000    0.45
       0.15   0.848710124     1.488927E+00    1.732820E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.58095    0.47510 0.40075 5.00000 7.00000 16.00000    36.00000    0.467548875
       0.20   0.829169096     1.496370E+00    1.826639E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.61000    0.50000 0.38000 5.00000 7.00000 16.00000    36.00000    0.48
       0.25   0.819497568     1.473785E+00    1.854356E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.62651    0.50000 0.37450 5.00000 7.00000 16.00000    36.00000    0.48
       0.30   0.814346585     1.455332E+00    1.877313E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.64000    0.50000 0.37000 5.00000 7.00000 16.00000    36.00000    0.48
       0.40   0.814243546     1.428722E+00    1.886767E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.61747    0.48874 0.37000 5.00000 7.00000 16.00000    36.00000    0.468736584
       0.50   0.812705538     1.405794E+00    1.841480E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.60000    0.48000 0.37000 5.00000 7.00000 16.00000    36.00000    0.46
       0.60   0.815801037     1.387061E+00    1.805286E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.58422    0.47211 0.37789 5.00000 7.00000 16.00000    36.00000    0.457369656
       0.70   0.820928683     1.371222E+00    1.775240E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.57087    0.46544 0.38456 5.00000 7.00000 16.00000    36.00000    0.455145732
       0.80   0.827376206     1.357502E+00    1.749618E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.55932    0.45966 0.39034 5.00000 7.00000 16.00000    36.00000    0.453219281
       0.90   0.831408127     1.345400E+00    1.727324E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.54912    0.45456 0.39544 5.00000 7.00000 16.00000    36.00000    0.451520031
       1.00   0.840799961     1.334574E+00    1.707623E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.54000    0.45000 0.40000 5.00000 7.00000 16.00000    36.00000    0.45
       1.25   0.855226821     1.168218E+00    1.437403E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.53797    0.43984 0.40000 5.00000 7.00000 16.00000    36.00000    0.441875439
       1.50   0.871104233     1.032296E+00    1.248681E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.53631    0.43155 0.40000 5.00000 7.00000 16.00000    36.00000    0.43523719
       2.00   0.891458727     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.53369    0.41845 0.40000 5.00000 7.00000 16.00000    36.00000    0.42476281
       2.50   0.903858490     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.53166    0.40830 0.40000 5.00000 7.00000 16.00000    36.00000    0.416638249
       3.00   0.913991280     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.53000    0.40000 0.40000 5.00000 7.00000 16.00000    36.00000    0.41
       4.00   0.913352473     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.53000    0.40000 0.40000 5.00000 7.00000 16.00000    36.00000    0.41
       5.00   0.912857283     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09   0.53000    0.40000 0.40000 5.00000 7.00000 16.00000    36.00000    0.41
       """)

COEFFS_FS_ROCK_SWISS03 = CoeffsTable(sa_damping=5, table="""\
       IMT    k_adj           a1              a2              b1              b2              Rm              phi_11      phi_21      C2          Mc1    Mc2    Rc11    Rc21  mean_phi_ss
       pga    1.037040000     1.642085E+00    1.833001E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.58        0.47        0.35        5      7      16      36    0.46
       0.05   1.152476093     1.590620E+00    1.617332E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.5520412   0.4490309   0.4059176   5      7      16      36    0.4530103
       0.10   0.995583662     1.478437E+00    1.608714E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.54        0.44        0.43        5      7      16      36    0.45
       0.15   0.948713303     1.488927E+00    1.732820E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.580947375 0.47509775  0.400751875 5      7      16      36    0.467548875
       0.20   0.936827687     1.496370E+00    1.826639E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.61        0.5         0.38        5      7      16      36    0.48
       0.25   0.941001497     1.473785E+00    1.854356E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.626510191 0.5         0.374496603 5      7      16      36    0.48
       0.30   0.951517574     1.455332E+00    1.877313E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.64        0.5         0.37        5      7      16      36    0.48
       0.40   0.980951997     1.428722E+00    1.886767E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.617473168 0.488736584 0.37        5      7      16      36    0.468736584
       0.50   0.999448607     1.405794E+00    1.841480E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.6         0.48        0.37        5      7      16      36    0.46
       0.60   1.013777169     1.387061E+00    1.805286E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.584217936 0.472108968 0.377891032 5      7      16      36    0.457369656
       0.70   1.022460327     1.371222E+00    1.775240E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.57087439  0.465437195 0.384562805 5      7      16      36    0.455145732
       0.80   1.026784122     1.357502E+00    1.749618E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.559315686 0.459657843 0.390342157 5      7      16      36    0.453219281
       0.90   1.024261640     1.345400E+00    1.727324E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.549120186 0.454560093 0.395439907 5      7      16      36    0.451520031
       1.00   1.025806874     1.334574E+00    1.707623E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.54        0.45        0.4         5      7      16      36    0.45
       1.25   1.014356561     1.168218E+00    1.437403E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.53796886  0.439844299 0.4         5      7      16      36    0.441875439
       1.50   1.006424520     1.032296E+00    1.248681E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.536309298 0.431546488 0.4         5      7      16      36    0.43523719
       2.00   0.990611915     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.533690702 0.418453512 0.4         5      7      16      36    0.42476281
       2.50   0.981034792     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.531659562 0.408297812 0.4         5      7      16      36    0.416638249
       3.00   0.978562884     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.53        0.4         0.4         5      7      16      36    0.41
       4.00   0.958470267     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.53        0.4         0.4         5      7      16      36    0.41
       5.00   0.943169770     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.53        0.4         0.4         5      7      16      36    0.41
       """)

COEFFS_FS_ROCK_SWISS08 = CoeffsTable(sa_damping=5, table="""\
       IMT    k_adj           a1              a2              b1              b2              Rm              phi_11      phi_21      C2          Mc1    Mc2    Rc11    Rc21  mean_phi_ss
       pga    1.414560000     1.642085E+00    1.833001E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.58        0.47        0.35        5      7      16      36    0.46
       0.05   2.012007281     1.590620E+00    1.617332E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.5520412   0.4490309   0.4059176   5      7      16      36    0.4530103
       0.10   1.363140802     1.478437E+00    1.608714E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.54        0.44        0.43        5      7      16      36    0.45
       0.15   1.143182969     1.488927E+00    1.732820E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.580947375 0.47509775  0.400751875 5      7      16      36    0.467548875
       0.20   1.039739290     1.496370E+00    1.826639E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.61        0.5         0.38        5      7      16      36    0.48
       0.25   0.983550465     1.473785E+00    1.854356E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.626510191 0.5         0.374496603 5      7      16      36    0.48
       0.30   0.948764154     1.455332E+00    1.877313E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.64        0.5         0.37        5      7      16      36    0.48
       0.40   0.913557081     1.428722E+00    1.886767E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.617473168 0.488736584 0.37        5      7      16      36    0.468736584
       0.50   0.891456331     1.405794E+00    1.841480E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.6         0.48        0.37        5      7      16      36    0.46
       0.60   0.881236402     1.387061E+00    1.805286E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.584217936 0.472108968 0.377891032 5      7      16      36    0.457369656
       0.70   0.877081048     1.371222E+00    1.775240E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.57087439  0.465437195 0.384562805 5      7      16      36    0.455145732
       0.80   0.876720492     1.357502E+00    1.749618E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.559315686 0.459657843 0.390342157 5      7      16      36    0.453219281
       0.90   0.875316976     1.345400E+00    1.727324E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.549120186 0.454560093 0.395439907 5      7      16      36    0.451520031
       1.00   0.880663480     1.334574E+00    1.707623E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.54        0.45        0.4         5      7      16      36    0.45
       1.25   0.887163571     1.168218E+00    1.437403E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.53796886  0.439844299 0.4         5      7      16      36    0.441875439
       1.50   0.897498847     1.032296E+00    1.248681E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.536309298 0.431546488 0.4         5      7      16      36    0.43523719
       2.00   0.910434478     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.533690702 0.418453512 0.4         5      7      16      36    0.42476281
       2.50   0.918285853     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.531659562 0.408297812 0.4         5      7      16      36    0.416638249
       3.00   0.926223751     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.53        0.4         0.4         5      7      16      36    0.41
       4.00   0.922398917     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.53        0.4         0.4         5      7      16      36    0.41
       5.00   0.919443026     8.178257E-01    1.000000E+00    1.000000E+00    0.000000E+00    1.000000E+09    0.53        0.4         0.4         5      7      16      36    0.41
       """)
