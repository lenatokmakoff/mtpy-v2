"""
Utilities for converting EDI data to MARE2DEM compatible format for use 
with MARE2DEM software. 

Adapted from MTgeophysics/mtpy/mtpy/modeling/mare2dem.py


Created on Fri Oct 4 13:57:49 2024
tokmakol@oregonstate.edu
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
from pathlib import Path
import pandas as pd
from loguru import logger

from mtpy.core.mt_dataframe import MTDataFrame
from mtpy.modeling.mare2dem import Mare2DData

# =============================================================================

class Mare2DData: 
    """Mare2DData will read EDI files and convert them to MARE2DEM format.
    
    :param edi_list: List of edi files to be converted.
    """
    
    def __init__(self, dataframe=None, center_point=None, **kwargs):
        self.logger = logger

        self.dataframe = dataframe

        if center_point is None:
            self.center_point = MTLocation()
        else:
            self.center_point = center_point

        self.wave_sign_impedance = "+"
        self.wave_sign_tipper = "+"
        self.z_units = "[mV/km]/[nT]"
        self.t_units = ""
        self.inv_mode = "1"
        self.formatting = "1"
        self.rotation_angle = 0

        self.z_model_error = ModelErrors(
            error_value=5,
            error_type="geometric_mean",
            floor=True,
            mode="impedance",
        )
        self.t_model_error = ModelErrors(
            error_value=0.02,
            error_type="absolute",
            floor=True,
            mode="tipper",
        )

        self.fn_basename = "ModEM_Data.dat"
        self.save_path = Path.cwd()

        self.topography = True

        self.inv_mode_dict = {
            "1": ["Full_Impedance", "Full_Vertical_Components"],
            "2": ["Full_Impedance"],
            "3": ["Off_Diagonal_Impedance", "Full_Vertical_Components"],
            "4": ["Off_Diagonal_Impedance"],
            "5": ["Full_Vertical_Components"],
            "6": ["Full_Interstation_TF"],
            "7": ["Off_Diagonal_Rho_Phase"],
            "8": ["Phase_Tensor"],
        }
        self.inv_comp_dict = {
            "Full_Impedance": ["z_xx", "z_xy", "z_yx", "z_yy"],
            "Off_Diagonal_Impedance": ["z_xy", "z_yx"],
            "Full_Vertical_Components": ["t_zx", "t_zy"],
            "Phase_Tensor": ["pt_xx", "pt_xy", "pt_yx", "pt_yy"],
        }

        self.header_string = " ".join(
            [
                "# Period(s)",
                "Code",
                "GG_Lat",
                "GG_Lon",
                "X(m)",
                "Y(m)",
                "Z(m)",
                "Component",
                "Real",
                "Imag",
                "Error",
            ]
        )

        self._df_keys = [
            "period",
            "station",
            "latitude",
            "longitude",
            "model_north",
            "model_east",
            "model_elevation",
            "comp",
            "real",
            "imag",
            "error",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        """Str function."""
        lines = ["ModEM Data Object:"]
        if self.dataframe is not None:
            lines += [
                f"\tNumber of impedance stations: {self.get_n_stations('impedance')}"
            ]
            lines += [
                f"\tNumber of tipper stations: {self.get_n_stations('vertical')}"
            ]
            lines += [
                f"\tNumber of phase tensor stations: {self.get_n_stations('phase_tensor')}"
            ]
            lines += [f"\tNumber of periods:  {self.n_periods}"]
            lines += ["\tPeriod range (s):  "]
            lines += [f"\t\tMin: {self.period.min():.5g}"]
            lines += [f"\t\tMax: {self.period.max():.5g}"]
            lines += [f"\tRotation angle:     {self.rotation_angle}"]
            lines += ["\tData center:        "]
            lines += [
                f"\t\tLatitude:  {self.center_point.latitude:>8.4f} deg "
                f"\tNorthing: {self.center_point.north:.4f} m"
            ]
            lines += [
                f"\t\tLongitude: {self.center_point.longitude:>8.4f} deg "
                f"\tEasting: {self.center_point.east:.4f} m"
            ]
            lines += [
                f"\t\tDatum epsg: {self.center_point.datum_epsg}"
                f"\t\t\tUTM epsg:   {self.center_point.utm_epsg}"
            ]
            lines += [f"\t\tElevation:  {self.center_point.elevation:.1f} m"]

            lines += [
                f"\tImpedance data:     {self.dataframe.z_xy.mean() != 0.0}"
            ]
            lines += [
                f"\tTipper data:        {self.dataframe.t_zx.mean() != 0.0}"
            ]
            lines += [
                f"\tInversion Mode:   {', '.join(self.inv_mode_dict[self.inv_mode])}"
            ]

        return "\n".join(lines)

    def __repr__(self):
        """Repr function."""
        return self.__str__()

    @property
    def dataframe(self):
        """Dataframe function."""
        return self._mt_dataframe.dataframe

    @dataframe.setter
    def dataframe(self, df):
        """Set dataframe to an MTDataframe.
        :param df: DESCRIPTION.
        :type df: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if df is None:
            self._mt_dataframe = MTDataFrame()

        elif isinstance(df, (pd.DataFrame, MTDataFrame, np.ndarray)):
            self._mt_dataframe = MTDataFrame(df)

        else:
            raise TypeError(
                f"Input must be a dataframe or MTDataFrame object not {type(df)}"
            )

        self._mt_dataframe.dataframe.reset_index(drop=True, inplace=True)

    @property
    def model_parameters(self):
        """Model parameters."""
        params = {
            "wave_sign_impedance": self.wave_sign_impedance,
            "wave_sign_tipper": self.wave_sign_tipper,
            "z_units": self.z_units,
            "t_units": self.t_units,
            "inv_mode": self.inv_mode,
            "formatting": self.formatting,
            "data_filename": self.data_filename,
            "topography": self.topography,
            "rotation_angle": self.rotation_angle,
            "center_point.latitude": self.center_point.latitude,
            "center_point.longitue": self.center_point.longitude,
            "center_point.elevation": self.center_point.elevation,
            "center_point.utm_epsg": self.center_point.utm_epsg,
            "center_point.datum_epsg": self.center_point.datum_epsg,
        }

        for key, value in self.z_model_error.error_parameters.items():
            params[f"z_model_error.{key}"] = value
        for key, value in self.t_model_error.error_parameters.items():
            params[f"t_model_error.{key}"] = value

        return params

    @property
    def data_filename(self):
        """Data filename."""
        return self.save_path.joinpath(self.fn_basename)

    @data_filename.setter
    def data_filename(self, value):
        """Data filename."""
        if value is not None:
            value = Path(value)
            if value.parent == Path("."):
                self.fn_basename = value.name
            else:
                self.save_path = value.parent
                self.fn_basename = value.name

    @property
    def period(self):
        """Period function."""
        if self.dataframe is not None:
            return np.sort(self.dataframe.period.unique())

    def get_n_stations(self, mode):
        """Get n stations."""
        if self.dataframe is not None:
            if "impedance" in mode.lower():
                return (
                    self.dataframe.loc[
                        (self.dataframe.z_xx != 0)
                        | (self.dataframe.z_xy != 0)
                        | (self.dataframe.z_yx != 0)
                        | (self.dataframe.z_yy != 0),
                        "station",
                    ]
                    .unique()
                    .size
                )
            elif "vertical" in mode.lower():
                return (
                    self.dataframe.loc[
                        (self.dataframe.t_zx != 0)
                        | (self.dataframe.t_zy != 0),
                        "station",
                    ]
                    .unique()
                    .size
                )
            elif "phase_tensor" in mode.lower():
                return (
                    self.dataframe.loc[
                        (self.dataframe.pt_xx != 0)
                        | (self.dataframe.pt_xy != 0)
                        | (self.dataframe.pt_yx != 0)
                        | (self.dataframe.pt_yy != 0),
                        "station",
                    ]
                    .unique()
                    .size
                )

    @property
    def n_periods(self):
        """N periods."""
        return self.period.size

    def _get_components(self):
        """Get components to write out."""

        comps = []
        for inv_modes in self.inv_mode_dict[self.inv_mode]:
            comps += self.inv_comp_dict[inv_modes]

        return comps

    def _get_header_string(self, error_type, error_value):
        """Create the header strings

        # Created using MTpy calculated egbert_floor error of 5% data rotated 0.0_deg
        clockwise from N.
        :param error_type: The method to calculate the errors.
        :type error_type: string
        :param error_value: Value of error or error floor.
        :type error_value: float
        :param rotation_angle: Angle data have been rotated by.
        :type rotation_angle: float
        """

        h_str = []
        if np.atleast_1d(error_type).ndim == 2:
            h_str = (
                f"# Creating_software: MTpy v2, "
                f"error: [{error_type[0, 0]}, {error_type[0, 1]}, "
                f"{error_type[1, 0]}, {error_type[1, 1]}], "
            )
        else:
            h_str = f"# Creating_software: MTpy v2, error: {error_type}, "

        if np.atleast_1d(error_value).ndim == 2:
            h_str += (
                f"error floors of {error_value[0, 0]:.0f}%, "
                f"{error_value[0, 1]:.0f}%, "
                f"{error_value[1, 0]:.0f}%, "
                f"{error_value[1, 1]:.0f}%, "
                f"data rotated {self.rotation_angle:.1f}_deg clockwise from N, "
                f"{self.center_point.utm_crs}"
            )

        else:
            if error_value > 1:
                fmt = ".0f"
                units = "%"
            elif error_value < 1:
                fmt = ".2f"
                units = ""
            h_str += (
                f"error_value: {error_value:{fmt}}{units}, data_rotation: "
                f"{self.rotation_angle:.1f} deg clockwise, "
                f"model_{self.center_point.utm_crs}"
            )

        return h_str

    
    def _write_comp(self, row, comp):
        """Write a single row.
        :param row: DESCRIPTION.
        :type row: TYPE
        :param comp: DESCRIPTION.
        :type comp: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        # need zxx instead of z_xx
        inv_comp = comp.replace("_", "", 1)
        if inv_comp[1].lower() == "z":
            inv_comp = inv_comp.replace("z", "")
        com = f"{inv_comp.upper():>8}"

        value = np.nan_to_num(getattr(row, comp))
        err = getattr(row, f"{comp}_model_error")

        if (
            value.real != 0.0
            and value.imag != 0.0
            and value.real != 1e32
            and value.imag != 1e32
        ):
            if self.formatting == "1":
                per = f"{row.period:<12.5e}"
                sta = f"{row.station:>7}"
                lat = f"{row.latitude:> 9.3f}"
                lon = f"{row.longitude:> 9.3f}"
                eas = f"{row.model_east:> 12.3f}"
                nor = f"{row.model_north:> 12.3f}"
                if self.topography:
                    ele = f"{row.model_elevation:> 12.3f}"
                else:
                    ele = f"{0:> 12.3f}"

                if self.z_units.lower() == "ohm":
                    rea = f"{value.real / 796.:> 14.6e}"
                    ima = f"{value.imag / 796.:> 14.6e}"
                elif self.z_units.lower() not in (
                    "[v/m]/[t]",
                    "[mv/km]/[nt]",
                ):
                    raise ValueError(f"Unsupported unit '{self.z_units}'")
                else:
                    rea = f"{value.real:> 14.6e}"
                    ima = f"{value.imag:> 14.6e}"

            elif self.formatting == "2":
                per = f"{row.period:<14.6e}"
                sta = f"{row.station:>10}"
                lat = f"{row.latitude:> 14.6f}"
                lon = f"{row.longitude:> 14.6f}"
                eas = f"{row.model_east:> 15.3f}"
                nor = f"{row.model_north:> 15.3f}"
                if self.topography:
                    ele = f"{row.model_elevation:> 10.3f}"
                else:
                    ele = f"{0:> 10.3f}"

                if self.z_units.lower() == "ohm":
                    rea = f"{value.real / 796.:> 17.6e}"
                    ima = f"{value.imag / 796.:> 17.6e}"
                elif self.z_units.lower() not in (
                    "[v/m]/[t]",
                    "[mv/km]/[nt]",
                ):
                    raise ValueError(f"Unsupported unit '{self.z_units}'")
                else:
                    rea = f"{value.real:> 17.6e}"
                    ima = f"{value.imag:> 17.6e}"

            else:
                raise NotImplementedError(
                    f"format {self.formatting} ({type(self.formatting)}) is "
                    "not supported."
                )

            if np.isinf(err) or np.isnan(err):
                err = 10 ** (
                    np.floor(np.log10(abs(max([float(rea), float(ima)]))))
                )
            abs_err = f"{err:> 14.6e}"

            return "".join(
                [
                    per,
                    sta,
                    lat,
                    lon,
                    nor,
                    eas,
                    ele,
                    com,
                    rea,
                    ima,
                    abs_err,
                ]
            )

    def _check_for_errors_of_zero(self):
        """Need to check for any zeros in the error values which can prevent
        ModEM from running.
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        ## check for zeros in model error
        for comp in ["z_xx", "z_xy", "z_yx", "z_yy", "t_zx", "t_zy"]:
            find_zeros = np.where(self.dataframe[f"{comp}_model_error"] == 0)[
                0
            ]
            find_zeros_data = np.where(self.dataframe[f"{comp}"] == 0)[0]

            if find_zeros.shape == find_zeros_data.shape:
                continue
            if find_zeros.shape[0] > 0:
                if comp in ["z_xx", "z_xy", "z_yx", "z_yy"]:
                    error_percent = self.z_model_error.error_value
                elif "t" in comp:
                    error_percent = self.t_model_error.error_value

                self.logger.warning(
                    f"Found errors with values of 0 in {comp} "
                    f"{len(find_zeros)} times. Setting error as {comp} x "
                    f"{error_percent}."
                )

                self.dataframe.loc[
                    find_zeros.tolist(), f"{comp}_model_error"
                ] = (
                    abs(self.dataframe[f"{comp}"].iloc[list(find_zeros)])
                    * error_percent
                )

    def _check_for_too_small_errors(self, tol=0.02):
        """Check for too small of errors relative to the error floor."""

        for comp in ["z_xx", "z_xy", "z_yx", "z_yy", "t_zx", "t_zy"]:
            find_small = np.where(
                self.dataframe[f"{comp}_model_error"]
                / abs(self.dataframe[comp])
                < tol
            )[0]
            if find_small.shape[0] > 0:
                if comp.startswith("z"):
                    error_percent = self.z_model_error.error_value
                elif comp.startswith("t"):
                    error_percent = self.t_model_error.error_value

                self.logger.warning(
                    f"Found errors with values less than {tol} in {comp} "
                    f"{len(find_small)} times. Setting error as {comp} x "
                    f"{error_percent}."
                )
                self.dataframe.loc[
                    find_small.tolist(), f"{comp}_model_error"
                ] = (
                    abs(self.dataframe[f"{comp}"].iloc[list(find_small)])
                    * error_percent
                )

    def _check_for_too_big_values(self, tol=1e10):
        """Check for too small of errors relative to the error floor."""

        for comp in ["z_xx", "z_xy", "z_yx", "z_yy", "t_zx", "t_zy"]:
            find_big = np.where(np.abs(self.dataframe[comp]) > tol)[0]
            if find_big.shape[0] > 0:
                self.logger.warning(
                    f"Found values in {comp} larger than {tol} "
                    f"{len(find_big)} times. Setting to nan"
                )
                self.dataframe.loc[find_big.tolist(), comp] = (
                    np.nan + 1j * np.nan
                )
                self.dataframe.loc[
                    find_big.tolist(), f"{comp}_model_error"
                ] = np.nan

    def _check_for_too_small_values(self, tol=1e-10):
        """Check for too small of errors relative to the error floor."""

        for comp in ["z_xx", "z_xy", "z_yx", "z_yy", "t_zx", "t_zy"]:
            find_small = np.where(np.abs(self.dataframe[comp]) < tol)[0]
            find_zeros_data = np.where(
                np.nan_to_num(self.dataframe[f"{comp}"]) == 0
            )[0]
            if find_small.shape == find_zeros_data.shape:
                continue
            if find_small.shape[0] > 0:
                self.logger.warning(
                    f"Found values in {comp} smaller than {tol} "
                    f"{len(find_small)} times. Setting to nan"
                )
                self.dataframe.loc[find_small.tolist(), comp] = (
                    np.nan + 1j * np.nan
                )
                self.dataframe.loc[
                    find_small.tolist(), f"{comp}_model_error"
                ] = np.nan
# # ============================================================================= LMT
def write_mare2dem_data(o2d_filepath, site_locations, site_elevations, site_names,
                        mare_origin, utm_zone, gstrike, solve_statics=False, savepath=None):
    """
    Uses an Occam2D data file and site locations + elevations to
    generate a MARE2DEM data file.

    Parameters
    ----------
    o2d_filepath : bytes or str
        Full path to the Occam2D data file created from EDI data.
        
    site_locations : np.ndarray
        Array of shape (nstations). According to the original comments
        in the `EDI2Mare2DEM_withOccam2D_new` script, this is the
        site elevation but in Mare2D coordinates, however it gets set
        as the 'y' component of receiver locations in the Mare2DEM
        data file.
        
    site_elevations : np.ndarray
        Array of shape (nstations). I think this is the site elevation
        in UTM coordinates. Gets set as the 'z' component of receiver
        locations in the Mare2DEM data file.
        
    site_names : np.ndarray
        Array of site_names.
        
    mare_origin : tuple
        Tuple of float (x, y, utm_zone). The Mare2D origin in UTM
        coordinates. Note according to the original script, Mare2D origin
        is the middle of the profile line. utm_zone is the UTM string,
        e.g. '54S'.
        
    gstrike : int
        The 2D strike, same as
        `geoelectric_strike` in Occam2D model. This information is used
        for the UTM origin line in the data file.
        
    solve_statics : bool or list, optional
        If boolean, sets whether to solve statics for all stations.
        A list of site names can be passed. Sites in the list will
        have solve_statics set to True, any sites not included
        are set to False. This writes a '1' or a '0' in
        the Receiver section of the Mare2D data file in the SovleStatics
        column.
        
    savepath : bytes or str, optional
        Full path of where to save the Mare2D data file. If not
        provided, will be saved as 'Mare2D_data.txt' in working
        directory.
        
    """
   
    # Prepare O2D data for data block
    o2d_sites = []
    o2d_freqs = []
    o2d_types = []
    o2d_datums = []
    o2d_errors = []
    
    with open(o2d_filepath, 'r') as f:
        read_data = f.readlines()
        reading_data = False
        
        for line in read_data:
            
            if line.startswith('SITE '):
                reading_data = True
                continue
            
            elif reading_data:
                parts = line.split()
                o2d_sites.append(parts[0])
                o2d_freqs.append(parts[1])
                o2d_types.append(parts[2])
                o2d_datums.append(parts[3])
                o2d_errors.append(parts[4])

    sites = np.array(o2d_sites, dtype=np.int8)
    freqs = np.array(o2d_freqs, dtype=np.int8)
    types = np.array(o2d_types, dtype=np.int8)
    datums = np.array(o2d_datums, dtype=np.float64)
    errors = np.array(o2d_errors, dtype=np.float64)
    
    # Convert occam2d types to mare2d types
    # The below is: for each element in types array, return corresponding element in conversion
    
    # dict, if not found in dict return original element
    type_conversion = {1: 123, 2: 104, 3: 133, 4: 134, 5: 125, 6: 106, 9: 103, 10: 105}
    types = np.vectorize(lambda x: type_conversion.get(x, x))(types)
    # Put into dataframe for easier stringifying
    # Note: TX# == RX# == site ID for MT stations
    data_df = pd.DataFrame((types, freqs, sites, sites, datums, errors), dtype=np.object).T
    # Bit of a hack: add the '!' to the data frame header because the 'type' integer is small
    # enough that the 'Type' header will have no left whitespace padding, so we can't prepend
    # it with '!' without throwing off the alignment.
    data_df.columns = ['! Type', 'Freq #', 'Tx #', 'Rx #', 'Data', 'StdErr']
    data_str = data_df.to_string(index=False, float_format=lambda x: '%.4f' % x)

    # Prepare data for the Reciever block
    # Zeros of shape (n_sites) for X (as float), Theta, Alpha, Beta and Length (ints) columns
    
    x_col = np.zeros(site_locations.shape, dtype=np.float64)
    zero_ints = np.zeros(site_locations.shape, dtype=np.int8)
    t_col, a_col, b_col, l_col = zero_ints, zero_ints, zero_ints, zero_ints
    # add 0.1 m (shift the sites 10 cm beneath subsurface as recommended)
    site_elevations += 0.1
   
    # According to original script, need to reread the Occam2D file to get stations in the correct
    # order
    if isinstance(solve_statics, bool):
        statics = np.ones(site_locations.shape, dtype=np.int8) if solve_statics else zero_ints
    else:
        statics = np.zeros(site_locations.shape)
        for sn in solve_statics:
            statics[np.where(site_names == sn)] = 1
   
    # Put into dataframe for easier stringifying
    recv_df = pd.DataFrame((x_col, site_locations, site_elevations, t_col, a_col, b_col, l_col,
                            statics, site_names)).T
    recv_df.columns = ['X', 'Y', 'Z', 'Theta', 'Alpha', 'Beta', 'Length', 'SolveStatic', 'Name']
    recv_str = list(recv_df.to_string(index=False, float_format=lambda x: '%.6f' % x))
    
    # Replace the first char of header with Mare2DEM comment symbol '!'
    # This way the header is correct but Pandas handles the alignment and spacing
    recv_str[0] = '!'
    recv_str = "".join(recv_str)

    o2d_data = o2d.Data()
    o2d_data.read_data_file(o2d_filepath)

    if savepath is None:
        savepath = os.path.join(os.getcwd(), 'Mare2D_data.txt')
   
    with open(savepath, 'w') as output:
        # 1. header
        fstring = 'Format:  EMData_2.2\n'
        fstring += 'UTM of x,y origin (UTM zone, N, E, 2D strike):'
        gstrike = float(gstrike)
        fstring += ' {:s}{:>13.1f}{:>13.1f}\t{:f}\n'.format(
            utm_zone, mare_origin[0], mare_origin[1], gstrike)

        # 2. frequencies
        fstring += '# MT Frequencies:    {}\n'.format(len(o2d_data.freq))
        fstring += '\n'.join([str(round(f, 8)) for f in o2d_data.freq])

        # 3. receiver info
        fstring += '\n# MT Receivers:      {}\n'.format(len(site_names))
        fstring += recv_str
        fstring += '\n'

        # 4. data
        fstring += '# Data:       {}\n'.format(len(datums))
        fstring += data_str

        output.write(fstring)
# =============================================================================
    def write_data_file(
        self,
        file_name=None,
        save_path=None,
        fn_basename=None,
        elevation=False,
    ):
        """Write data file.
        :param file_name:
            Defaults to None.
        :param save_path: Full directory to save file to, defaults to None.
        :type save_path: string or Path, optional
        :param fn_basename: Basename of the saved file, defaults to None.
        :type fn_basename: string, optional
        :param elevation: If True adds in elevation from 'rel_elev' column in data
            array, defaults to False.
        :type elevation: boolean, optional
        :raises NotImplementedError: If the inversion mode is not supported.
        :raises ValueError: :class:`mtpy.utils.exceptions.ValueError` if a parameter.
        :return: Full path to data file.
        :rtype: Path
        """

        if self.dataframe is None:
            raise ValueError(
                "A DataFrame needs to be present to write a ModEM data file"
            )

        if file_name is not None:
            self.data_filename = file_name

        if save_path is not None:
            self.save_path = Path(save_path)
        if fn_basename is not None:
            self.data_filename = fn_basename

        self._check_for_errors_of_zero()
        self._check_for_too_small_errors()
        self._check_for_too_big_values()
        self._check_for_too_small_values()

        z_lines = []
        t_lines = []
        for inv_mode in self.inv_mode_dict[self.inv_mode]:
            if "impedance" in inv_mode.lower():
                z_lines = self._write_header(inv_mode)

            elif "vertical" in inv_mode.lower():
                t_lines = self._write_header(inv_mode)

            else:
                # maybe error here
                raise NotImplementedError(
                    f"inv_mode {inv_mode} is not supported yet"
                )

        comps = self._get_components()
        # Iterate over stations and sort by period
        for station in self.dataframe.station.unique():
            sdf = self.dataframe.loc[self.dataframe.station == station]
            sdf.sort_values("period")

            for row in sdf.itertuples():
                for comp in comps:
                    d_line = self._write_comp(row, comp)
                    if d_line is None:
                        continue

                    if comp.startswith("z"):
                        z_lines.append(d_line)
                    elif comp.startswith("t"):
                        t_lines.append(d_line)

        with open(self.data_filename, "w") as dfid:
            dfid.write("\n".join(z_lines + t_lines))

        self.logger.info(
            "Wrote ModEM data file to {0}".format(self.data_filename)
        )
        return self.data_filename

    def _read_header(self, header_lines):
        """Read header lines.
        :param header_lines: DESCRIPTION.
        :type header_lines: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        mode = None
        inv_list = []
        header_list = []
        metadata_list = []
        n_periods = 0
        n_stations = 0
        self.center_point = MTLocation()
        for hline in header_lines:
            if hline.find("#") == 0:
                if "period" not in hline.lower():
                    header_list.append(hline.strip())
            elif hline.find(">") == 0:
                # modem outputs only 7 characters for the lat and lon
                # if there is a negative they merge together, need to split
                # them up
                hline = hline.replace("-", " -")
                metadata_list.append(hline[1:].strip())
                if hline.lower().find("ohm") > 0:
                    self.z_units = "ohm"
                    continue
                elif hline.lower().find("mv") > 0:
                    self.z_units = "[mV/km]/[nT]"
                    continue
                elif hline.lower().find("vertical") > 0:
                    mode = "vertical"
                    inv_list.append("Full_Vertical_Components")
                    continue
                elif hline.lower().find("impedance") > 0:
                    mode = "impedance"
                    inv_list.append("Full_Impedance")
                    continue

                if hline.find("exp") > 0:
                    if mode in ["impedance"]:
                        self.wave_sign_impedance = hline[hline.find("(") + 1]
                    elif mode in ["vertical"]:
                        self.wave_sign_tipper = hline[hline.find("(") + 1]

                elif (
                    len(hline[1:].strip().split()) >= 2
                    and hline.count(".") > 0
                ):
                    value_list = [
                        float(value) for value in hline[1:].strip().split()
                    ]
                    if value_list[0] != 0.0:
                        self.center_point.latitude = value_list[0]
                    if value_list[1] != 0.0:
                        self.center_point.longitude = value_list[1]
                    try:
                        self.center_point.elevation = value_list[2]
                    except IndexError:
                        self.center_point.elevation = 0.0
                        self.logger.debug(
                            "Did not find center elevation in data file"
                        )
                elif len(hline[1:].strip().split()) < 2:
                    try:
                        self.rotation_angle = float(hline[1:].strip())
                    except ValueError:
                        continue
                elif len(hline[1:].strip().split()) == 2:
                    n_periods = int(hline[1:].strip().split()[0])
                    n_stations = int(hline[1:].strip().split()[1])

        for head_line, inv_mode in zip(header_list, inv_list):
            self._parse_header_line(head_line, inv_mode)

        self._get_inv_mode(inv_list)

        return n_periods, n_stations

    def _parse_header_line(self, header_line, mode):
        """Parse header line."""

        if header_line == self.header_string:
            return

        item_dict = {
            "error": "error_type",
            "error_value": "error_value",
            "data_rotation": "rotation_angle",
            "model_epsg": "center_point.utm_epsg",
        }

        if header_line.count(",") > 0:
            header_list = header_line.split(",")
        else:
            header_list = header_line.split()

        if "impedance" in mode.lower():
            obj = self.z_model_error

        elif "vertical" in mode.lower():
            obj = self.t_model_error

        for ii, item in enumerate(header_list):
            item = item.lower()
            if item.count(":") > 0:
                item_list = [k.strip() for k in item.split(":")]
                if len(item_list) == 2:
                    key = item_list[0]
                    value = item_list[1].replace("%", "").split()[0]
                    if key in ["error_value", "data_rotation"]:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    try:
                        if key in ["model_epsg"]:
                            setattr(self.center_point, "utm_epsg", value)
                        elif "error" in key:
                            setattr(
                                obj,
                                item_dict[key],
                                value,
                            )
                        else:
                            setattr(self, item_dict["key"], value)
                    except KeyError:
                        continue

            ## Older files
            else:
                if item in ["calculated"]:
                    value = header_list[ii + 1]

                    if "floor" in value:
                        setattr(obj, "floor", True)
                        value = value.replace("_floor", "")
                    setattr(obj, "error_type", value)

                if item in ["of"]:
                    value = float(header_list[ii + 1].replace("%", ""))
                    setattr(obj, item_dict["error_value"], value)

                if "deg" in item:
                    setattr(
                        self,
                        item_dict["data_rotation"],
                        float(item.split("_")[0]),
                    )

    def _get_rotation_angle(self, header_line):
        """Get rotation angle."""
        # try to find rotation angle
        h_list = header_line.split()
        for hh, h_str in enumerate(h_list):
            if h_str.find("_deg") > 0:
                try:
                    self.rotation_angle = float(h_str[0 : h_str.find("_deg")])
                except ValueError:
                    pass

    def _get_inv_mode(self, inv_list):
        """Get inv mode."""
        # find inversion mode
        for inv_key in list(self.inv_mode_dict.keys()):
            inv_mode_list = self.inv_mode_dict[inv_key]
            if len(inv_mode_list) != inv_list:
                continue
            else:
                tf_arr = np.zeros(len(inv_list), dtype=bool)

                for tf, data_inv in enumerate(inv_list):
                    if data_inv in self.inv_mode_dict[inv_key]:
                        tf_arr[tf] = True

                if np.alltrue(tf_arr):
                    self.inv_mode = inv_key
                    break

    def _read_line(self, line):
        """Read a single line.
        :param line: DESCRIPTION.
        :type line: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        line_dict = dict(
            [(key, value) for key, value in zip(self._df_keys, line.split())]
        )
        for key in [
            "period",
            "latitude",
            "longitude",
            "model_east",
            "model_north",
            "model_elevation",
            "real",
            "imag",
            "error",
        ]:
            line_dict[key] = float(line_dict[key])

        comp = line_dict.pop("comp").lower()
        if comp.startswith("t"):
            comp = comp.replace("t", "tz")

        line_dict[f"{comp}_real"] = line_dict.pop("real")
        line_dict[f"{comp}_imag"] = line_dict.pop("imag")
        line_dict[f"{comp}_model_error"] = line_dict.pop("error")
        if line_dict[f"{comp}_model_error"] > 1e10:
            line_dict[f"{comp}_model_error"] = np.nan

        return line_dict

    def _change_comp(self, comp):
        """Change to z_, t_, pt_."""
        if comp.startswith("z"):
            comp = comp.replace("z", "z_")
        elif comp.startswith("t"):
            comp = comp.replace("t", "t_")
        elif comp.startswith("pt"):
            comp = comp.replace("pt", "pt_")
        return comp

    def read_data_file(self, data_fn):
        """Read data file.
        :param data_fn: Full path to data file name.
        :type data_fn: string or Path
        :raises ValueError: If cannot compute component.
        """

        self.data_filename = Path(data_fn)

        if self.data_filename is None:
            raise ValueError("data_fn is None, enter a data file to read.")
        elif not self.data_filename.is_file():
            raise ValueError(
                "Could not find {0}, check path".format(self.data_filename)
            )

        self.center_point = MTLocation()

        # open file get lines
        with open(self.data_filename, "r") as dfid:
            dlines = dfid.readlines()

        # read header information
        n_periods, n_stations = self._read_header(
            [line for line in dlines if ">" in line or "#" in line]
        )

        # create a list of dictionaries to make into a pandas dataframe
        entries = []
        for dline in dlines:
            if "#" in dline or ">" in dline:
                continue

            elif len(dline.split()) == len(self._df_keys):
                line_dict = self._read_line(dline)
                entries.append(line_dict)

        full_df = pd.DataFrame(entries)

        # group by period and station so that there is 1 row per period per station
        combined_df = full_df.groupby(
            ["station", "period"], as_index=False
        ).first()

        # combine real and imaginary
        cols = [c.split("_")[0] for c in combined_df.columns if "real" in c]
        for col in cols:
            combined_df[col] = (
                combined_df[f"{col}_real"] + 1j * combined_df[f"{col}_imag"]
            )
            combined_df.drop(
                [f"{col}_real", f"{col}_imag"], axis=1, inplace=True
            )

        combined_df["elevation"] = -1 * combined_df["model_elevation"]

        combined_df = self._rename_columns(combined_df)

        return MTDataFrame(combined_df)

    def _rename_columns(self, df):
        """Need to rename columns to z_, t_, pt_."""

        col_dict = {}
        for col in df.columns:
            if col.startswith("z"):
                col_dict[col] = col.replace("z", "z_")
            elif col.startswith("t"):
                col_dict[col] = col.replace("t", "t_")
            elif col.startswith("pt"):
                col_dict[col] = col.replace("pt", "pt_")
            else:
                continue

        df = df.rename(columns=col_dict)
        return df

    def fix_data_file(self, fn=None, n=3):
        """A newer compiled version of Modem outputs slightly different headers
        This aims to convert that into the older format
        :param fn: DESCRIPTION, defaults to None.
        :type fn: TYPE, optional
        :param n: DESCRIPTION, defaults to 3.
        :type n: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if fn:
            self.data_filename = Path(fn)
        with self.data_filename.open() as fid:
            lines = fid.readlines()

        def fix_line(line_list):
            """Fix line."""
            return (
                " ".join("".join(line_list).replace("\n", "").split()) + "\n"
            )

        h1 = fix_line(lines[0:n])
        h2 = fix_line(lines[n : 2 * n])

        find = None
        for index, line in enumerate(lines[2 * n + 1 :], start=2 * n + 1):
            if line.find("#") >= 0:
                find = index
                break

        if find is not None:
            h3 = fix_line(lines[find : find + n])
            h4 = fix_line(lines[find + n : find + 2 * n])

            new_lines = (
                [h1, h2]
                + lines[2 * n : find]
                + [h3, h4]
                + lines[find + 2 * n :]
            )
        else:
            new_lines = [h1, h2] + lines[2 * n :]

        with self.data_filename.open("w") as fid:
            fid.writelines(new_lines)

        return self.data_filename
