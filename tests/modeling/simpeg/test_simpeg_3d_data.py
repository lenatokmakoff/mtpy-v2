# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:09:02 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
import numpy as np
from mtpy import MTData

from mtpy_data import PROFILE_LIST
from mtpy.modeling.simpeg.data_3d import Simpeg3DData
from simpeg.electromagnetics.natural_source.survey import Data

# =============================================================================


class TestSimpeg3DData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.maxDiff = None
        self.md = MTData()
        self.md.add_station(
            [fn for fn in PROFILE_LIST if fn.name.startswith("16")]
        )
        # australian epsg
        self.md.utm_epsg = 4462

        # interpolate onto a common period range
        self.new_periods = np.logspace(-5, 1, 10)
        self.md.interpolate(self.new_periods, inplace=True, bounds_error=False)

        self.mt_df = self.md.to_dataframe()

    def setUp(self):
        self.simpeg_data = Simpeg3DData(self.mt_df)

    # def test_get_locations_fail(self):
    #     df = self.md.to_dataframe()
    #     df.profile_offset = 0
    #     s = Simpeg3DData(df)
    #     self.assertRaises(ValueError, getattr, s, "station_locations")

    def test_station_locations(self):
        with self.subTest("shape"):
            self.assertEqual((6, 3), self.simpeg_data.station_locations.shape)

        with self.subTest("easting"):
            self.assertListEqual(
                sorted(
                    np.round(
                        [
                            1748272.6633829,
                            1748748.38773332,
                            1749294.26095771,
                            1749782.36727311,
                            1750256.42072474,
                            1750746.91502363,
                        ],
                        5,
                    ).tolist()
                ),
                sorted(
                    np.round(
                        self.simpeg_data.station_locations[:, 0], 5
                    ).tolist()
                ),
            )

        with self.subTest("northing"):
            self.assertListEqual(
                sorted(
                    np.round(
                        [
                            392386.55064591,
                            392327.27572739,
                            392236.78815949,
                            392163.46012645,
                            392092.04770295,
                            392015.27551658,
                        ],
                        5,
                    ).tolist()
                ),
                sorted(
                    np.round(
                        self.simpeg_data.station_locations[:, 1], 5
                    ).tolist()
                ),
            )

        with self.subTest("elevation"):
            self.assertTrue(
                np.allclose(
                    self.simpeg_data.station_locations[:, 2],
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                )
            )

    def test_to_rec_array(self):
        survey_data = Data.fromRecArray(self.simpeg_data.to_rec_array())
        with self.subTest("survey frequencies"):
            self.assertTrue(
                np.allclose(
                    np.sort(self.simpeg_data.frequencies),
                    np.sort(survey_data.survey.frequencies[::-1]),
                )
            )

    def test_frequencies(self):
        self.assertTrue(
            np.allclose(1.0 / self.new_periods, self.simpeg_data.frequencies)
        )


#     def test_survey_te(self):
#         # simpeg sorts in order of lowest frequency to highest
#         with self.subTest("frequencies"):
#             self.assertTrue(
#                 np.allclose(
#                     1.0 / self.new_periods[::-1],
#                     self.simpeg_data.survey_te.frequencies,
#                 )
#             )

#     def test_survey_tm(self):
#         with self.subTest("frequencies"):
#             self.assertTrue(
#                 np.allclose(
#                     1.0 / self.new_periods[::-1],
#                     self.simpeg_data.survey_tm.frequencies,
#                 )
#             )

#     def test_te_observations(self):
#         with self.subTest("size"):
#             self.assertEqual(
#                 self.simpeg_data.te_observations.size,
#                 2
#                 * self.simpeg_data.n_frequencies
#                 * self.simpeg_data.n_stations,
#             )

#     def test_tm_observations(self):
#         with self.subTest("size"):
#             self.assertEqual(
#                 self.simpeg_data.tm_observations.size,
#                 2
#                 * self.simpeg_data.n_frequencies
#                 * self.simpeg_data.n_stations,
#             )

#     def test_te_data_errors(self):
#         with self.subTest("size"):
#             self.assertEqual(
#                 self.simpeg_data.te_data_errors.size,
#                 2
#                 * self.simpeg_data.n_frequencies
#                 * self.simpeg_data.n_stations,
#             )

#     def test_tm_data_errors(self):
#         with self.subTest("size"):
#             self.assertEqual(
#                 self.simpeg_data.tm_data_errors.size,
#                 2
#                 * self.simpeg_data.n_frequencies
#                 * self.simpeg_data.n_stations,
#             )


# class TestSimpeg2DRecipe(unittest.TestCase):
#     @classmethod
#     def setUpClass(self):
#         self.md = MTData()
#         self.md.add_station(
#             [fn for fn in PROFILE_LIST if fn.name.startswith("16")]
#         )
#         # australian epsg
#         self.md.utm_epsg = 4462

#         # extract profile
#         self.profile = self.md.get_profile(
#             149.15, -22.3257, 149.20, -22.3257, 1000
#         )
#         # interpolate onto a common period range
#         self.new_periods = np.logspace(-3, 0, 4)
#         self.profile.interpolate(
#             self.new_periods, inplace=True, bounds_error=False
#         )

#         self.mt_df = self.profile.to_dataframe()

#         self.simpeg_inversion = Simpeg2D(self.mt_df)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
