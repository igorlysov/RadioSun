from abc import ABCMeta, abstractmethod
from urllib.request import urlopen
from collections import OrderedDict
from datetime import datetime
import astropy.io.ascii
from astropy.table import Column, MaskedColumn, vstack, Table
from astropy.io import fits
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.signal import fftconvolve
from scipy.signal import find_peaks
from collections import defaultdict
import numpy as np
from functools import lru_cache
import re
from radiosun.scrapper import Scrapper
from radiosun.time import TimeRange
from radiosun.utils import *


class BaseClient(metaclass=ABCMeta):
    @abstractmethod
    def acquire_data():
        pass

    @abstractmethod
    def form_data():
        pass

    @abstractmethod
    def get_data():
        pass


class SRSClient(BaseClient):
    base_url = 'ftp://ftp.ngdc.noaa.gov/STP/swpc_products/daily_reports/solar_region_summaries/%Y/%m/%Y%m%dSRS.txt'

    def extract_lines(self, content):
        section, final_section = [], []

        for i, line in enumerate(content):
            if re.match(r'^(I\.|IA\.|II\.)', line):
                section.append(i)
            if re.match(r'^(III|COMMENT|EFFECTIVE 2 OCT 2000|PLAIN|This message is for users of the NOAA/SEC Space|NNN)', line, re.IGNORECASE):
                final_section.append(i)

        if final_section and final_section[0] > section[-1]:
            section.append(final_section[0])
        header = content[:section[0]] + [content[s] for s in section]
        for line in section:
            content[line] = '# ' + content[line]

        table1 = content[section[0]:section[1]]
        table1[1] = re.sub(r'Mag\s*Type', r'Magtype', table1[1], flags=re.IGNORECASE)
        table2 = content[section[1]:section[2]]
        if len(section) > 3:
             table3 = content[section[2]:section[3]]
             extra_lines = content[section[3]:]
        else:
            table3 = content[section[2]:]
            extra_lines = None
        data = [table1, table2, table3]
        for i, table in enumerate(data):
            if len(table) > 2 and table[2].strip().title() == 'None':
                del table[2]
        return header, data, extra_lines

    def proccess_lines(self, date, key, lines):
        column_mapping = {
            'Nmbr': 'Number',
            'Location': 'Location',
            'Lo': 'Carrington Longitude',
            'Area': 'Area',
            'Z': 'Z',
            'Ll': 'Longitudinal Extent',
            'Nn': 'Number of Sunspots',
            'Magtype': 'Mag Type',
            'Lat': 'Lat'
        }

        column_types = {
            'Number': np.dtype('i4'),
            'Location': np.dtype('U6'),
            'Carrington Longitude': np.dtype('i8'),
            'Area': np.dtype('i8'),
            'Z': np.dtype('U3'),
            'Longitudinal Extent': np.dtype('i8'),
            'Number of Sunspots': np.dtype('i8'),
            'Mag Type': np.dtype('S4'),
            'Lat': np.dtype('i8'),
        }

        if lines:
            raw_data = astropy.io.ascii.read(lines)
            column_names = list(raw_data.columns)
            raw_data.rename_columns(
                column_names, new_names=[column_mapping[col.title()] for col in column_names]
            )

            if len(raw_data) == 0:
                for c in raw_data.itercols():
                    c.dtype = column_types[c._name]
                raw_data.add_column(Column(data=None, name="ID", dtype=('S2')), index=0)
                raw_data.add_column(Column(data=None, name="Date", dtype=('S10')), index=0)
            else:
                raw_data.add_column(Column(data=[key] * len(raw_data), name="ID"), index=0)
                raw_data.add_column(Column(data=[date] * len(raw_data), name="Date"), index=0)
            return raw_data
        return None

    def parse_longitude(self, value):
        longitude_sign = {'W': 1, 'E': -1}
        if "W" in value or "E" in value:
            return longitude_sign[value[3]] * float(value[4:])

    def parse_latitude(self, value):
        latitude_sign = {'N': 1, 'S': -1}
        if "N" in value or "S" in value:
            return latitude_sign[value[0]] * float(value[1:3])

    def parse_location(self, column):
        latitude = MaskedColumn(name='Latitude')
        longitude = MaskedColumn(name='Longitude')

        for i, loc in enumerate(column):
            if loc:
                lat_value = self.parse_latitude(loc)
                long_val = self.parse_longitude(loc)
                latitude = latitude.insert(i, lat_value)
                longitude = longitude.insert(i, long_val)
            else:
                latitude = latitude.insert(i, None, mask=True)
                longitude = longitude.insert(i, None, mask=True)
        return latitude, longitude

    def parse_lat_col(self, column, latitude_column):
        for i, loc in enumerate(column):
            if loc:
                latitude_column.mask[i] = False
                latitude_column[i] = self.parse_latitude(loc)
        return latitude_column

    def acquire_data(self, timerange):
        scrapper = Scrapper(self.base_url)
        return scrapper.form_fileslist(timerange)

    def form_data(self, file_urls):
        total_table, section_lines, final_section_lines = [], [], []
        for file_url in file_urls:
            tables = []
            with urlopen(file_url) as response:
                content = response.read().decode('utf-8').split('\n')
                header, section_lines, supplementary_lines = self.extract_lines(content)
                issued_lines = [line for line in header if 'issued' in line.lower() and line.startswith(':')][0]
                _, date_text = issued_lines.strip().split(':')[1:]
                issued_date = datetime.strptime(date_text.strip(), "%Y %b %d %H%M UTC")
                meta_id = OrderedDict()
                for h in header:
                    if h.startswith(("I.", "IA.", "II.")):
                        pos = h.find('.')
                        id = h[:pos]
                        id_text = h[pos + 2:]
                        meta_id[id] = id_text.strip()

                for key, lines in zip(list(meta_id.keys()), section_lines):
                    raw_data = self.proccess_lines(issued_date.strftime("%Y-%m-%d"), key, lines)
                    tables.append(raw_data)
                stacked_table = vstack(tables)

                if 'Location' in stacked_table.columns:
                    col_lat, col_lon = self.parse_location(stacked_table['Location'])
                    del stacked_table['Location']
                    stacked_table.add_column(col_lat)
                    stacked_table.add_column(col_lon)

                if 'Lat' in stacked_table.columns:
                    self.parse_lat_col(stacked_table['Lat'], stacked_table['Latitude'])
                    del stacked_table['Lat']

            total_table.append(stacked_table)
        return Table(vstack(total_table))

    def get_data(self, timerange):
        file_urls = self.acquire_data(timerange)
        return self.form_data(file_urls)
    

class RATANClient(BaseClient):
    base_url = 'http://spbf.sao.ru/data/ratan/%Y/%m/%Y%m%d_%H%M%S_sun+0_out.fits'
    regex_pattern = '((\d{6,8})[^0-9].*[^0-9]0_out.fits)'

    convolution_template = pd.read_excel('radiosun/client/quiet_sun_template.xlsx')
    quiet_sun_model = pd.read_excel('radiosun/client/quiet_sun_model.xlsx')

    def condition(self, year, month, data_match):
        if int(year) < 2010 or (int(year) == 2010 and int(month) < 5):
            return f'{year[:2]}{data_match[:-4]}-{data_match[-4:-2]}-{data_match[-2:]}'
        else:
            return f'{data_match[:-4]}-{data_match[-4:-2]}-{data_match[-2:]}'

    def filter(self, url_list):
        pass

    @lru_cache(maxsize=None)
    def convolve_sun(self, sigma_horiz, sigma_vert, R):
        size = 1000
        x = np.linspace(-size//2, size//2, size)
        y = np.linspace(-size//2, size//2, size)
        gaussian = gauss2d(x, y, 1, 1, 0, 0, sigma_horiz, sigma_vert)
        rectangle = create_rectangle(size, 6 * sigma_horiz, 4 * R)
        sun_model = create_sun_model(size, R)
        # Perform convolutions
        convolved_gaussian = fftconvolve(sun_model, gaussian, mode='same', axes=1) / np.sum(gaussian)
        convolved_rectangle = fftconvolve(sun_model, rectangle, mode='same', axes=1) / np.sum(rectangle)

        convolved_gaussian = convolved_gaussian / np.max(convolved_gaussian)
        convolved_rectangle = convolved_rectangle / np.max(convolved_rectangle)

        conv_g = np.sum(convolved_gaussian, axis=0)
        conv_r = np.sum(convolved_rectangle, axis=0)
        # Calculate areas under the curve
        area_gaussian = calculate_area(conv_g)
        area_rectangle = calculate_area(conv_r)
        # Division of areas
        area_ratio = area_gaussian / area_rectangle
        return area_ratio

    def antenna_efficiency(self, freq, R):
        areas = []
        for f in freq:
            lambda_value = (3 * 10 ** 8) / (f * 10 ** 9) * 1000
            FWHM_h = 0.85 * lambda_value
            FWHM_v = 0.75 * lambda_value * 60

            sigma_h = round(bwhm_to_sigma(FWHM_h), 9)
            sigma_v = round(bwhm_to_sigma(FWHM_v), 9)

            area_ratio = self.convolve_sun(sigma_h, sigma_v, R)
            areas.append(area_ratio)
        return areas

    def calibrate_QSModel(self, x, scan_data, solar_r, frequency, flux_eficiency):
        K_b = 1.38 * 10 ** (-23) # Константа Больцамана
        c = 3 * 10 ** 8 # Скорость света в вакууме
        freq_size = len(frequency)
        model_freq = self.convolution_template.columns[1:].values.astype(float)

        template_freq = self.quiet_sun_model['freq']
        template_val = self.quiet_sun_model['T_brightness']
        real_brightness = interp1d(template_freq, template_val, bounds_error=False, fill_value="extrapolate")
        full_flux = np.column_stack([
            frequency,
            2 * 10 ** 22 * K_b * real_brightness(frequency) * SunSolidAngle(solar_r) * (c / (frequency * 10 ** 9)) ** (-2)
        ])

        R = scan_data[:, 0, :] + scan_data[:, 1, :]
        L = scan_data[:, 0, :] - scan_data[:, 1, :]

        columns = self.convolution_template.columns.values[1:].astype(float)
        mask = (x >= -1.0 * solar_r) & (x <= 1.0 * solar_r)

        calibrated_R = np.zeros((freq_size, R.shape[1]), dtype=float)
        calibrated_L = np.zeros((freq_size, L.shape[1]), dtype=float)
        theoretical_new = np.zeros((freq_size, L.shape[1]), dtype=float)

        for freq_num in range(freq_size):
            real_R, real_L = R[freq_num], L[freq_num]
            freq_diff = np.abs(columns - frequency[freq_num])
            freq_template = np.argmin(freq_diff) + 1

            template_values = self.convolution_template.iloc[:, freq_template].values.copy()
            x_values = self.convolution_template.iloc[:, 0].values.copy()

            convolution_template_arcsec = flip_and_concat(template_values)
            x_arcsec = flip_and_concat(x_values, flip_values=True)

            coeff = full_flux[freq_num, 1] * flux_eficiency[freq_num] / trapezoid(convolution_template_arcsec, x_arcsec)
            theoretical_data = np.interp(x, x_arcsec, coeff * convolution_template_arcsec)
            theoretical_new[freq_num] = theoretical_data

            res_R = minimize(error, np.array([1]), args=(real_R[mask], theoretical_data[mask]))
            res_L = minimize(error, np.array([1]), args=(real_L[mask], theoretical_data[mask]))

            calibrated_R[freq_num, :] = real_R * res_R.x
            calibrated_L[freq_num, :] = real_L * res_L.x
        return mask, (calibrated_R + calibrated_L) / 2, (calibrated_R - calibrated_L) / 2

    def heliocentric_transform(self, Lat, Long, SOLAR_R, SOLAR_B):
        return (
            SOLAR_R * np.cos(Lat * np.pi / 180) * np.sin(Long * np.pi / 180),
            SOLAR_R * (np.sin(Lat * np.pi / 180) * np.cos(SOLAR_B * np.pi / 180) - np.cos(Lat * np.pi / 180) * np.cos(Long * np.pi / 180) * np.sin(SOLAR_B * np.pi / 180))
        )

    def pozitional_rotation(self, Lat, Long, angle):
        return (
            Lat * np.cos(angle * np.pi / 180) - Long * np.sin(angle * np.pi / 180),
            Lat * np.sin(angle * np.pi / 180) + Long * np.cos(angle * np.pi / 180)
        )

    def differential_rotation(self, Lat):
        A = 14.713
        B = -2.396
        C = -1.787
        return A + B * np.sin(Lat * np.pi / 180) ** 2 + C * np.sin(Lat * np.pi / 180) ** 4

    def pozitional_angle(self, AZIMUTH, SOL_DEC, SOLAR_P):
        q = -np.arcsin(np.tan(AZIMUTH * np.pi / 180) * np.tan(SOL_DEC * np.pi / 180)) * 180 / np.pi
        p = SOLAR_P + 360.0 if np.abs(SOLAR_P) > 30 else SOLAR_P
        return (p + q)

    def active_regions_search(self, srs, x, V, mask):
        x = x[mask]
        V = np.sum(V, axis=0)[mask]

        wavelet = 'sym6'  # Daubechies wavelet
        level = 4  # Level of decomposition
        denoised_data = wavelet_denoise(V, wavelet, level)
        height_threshold = lambda x: np.abs(np.median(x) + 0.1 * np.std(x))
        # Finding peaks in the denoised data
        peaks, _ = find_peaks(denoised_data, height=height_threshold(denoised_data))
        valleys, _ = find_peaks(-denoised_data, height=height_threshold(-denoised_data))
        extremums = np.concatenate((peaks, valleys))

        theoretical_latitudes = np.array(srs['Latitude'])
        experimental_latitudes = x[extremums]
        abs_diff = np.abs(experimental_latitudes[:, np.newaxis] - theoretical_latitudes)
        min_index = np.argmin(abs_diff, axis=1)
        closest_data = srs[min_index]
        closest_data['Latitude'] = experimental_latitudes

        original_indices = np.where(mask)[0]
        extremums_original = original_indices[extremums]
        closest_data.add_column(Column(name='Data Index', data=extremums_original, dtype=('i4')), index=3)
        closest_data.add_column(Column(name='Masked Index', data=extremums, dtype=('i4')), index=4)
        return closest_data

    def make_multigauss_fit(self, x, y, peak_info):
        min_lat = np.min(peak_info['Latitude'])
        max_lat = np.max(peak_info['Latitude'])
        indexes = peak_info['Data Index']
        mask = (x >= min_lat - 50) & (x <= max_lat + 50)
        x_masked, y_masked = x[mask], y[mask]
        y_min = np.min(y_masked)
        ar_info = [[y[index] - y_min, x[index]] for index in indexes]
        widths = np.repeat(1, len(peak_info))
        initial_guesses = np.ravel(np.column_stack((ar_info, widths)))
        params, _ = leastsq(gaussian_mixture, initial_guesses, args=(x_masked, y_masked - y_min))
        return np.array(params), y_min, x_masked

    def gauss_analysis(self, x, scan_data, ar_info):
        ar_number = np.unique(ar_info['Number'])
        for noaa_ar in ar_number:
            region_info = ar_info[ar_info['Number'] == noaa_ar]
            indices = np.where(ar_info['Number'] == noaa_ar)[0]
            for index, elem in enumerate(scan_data):
                freq, I_data, V_data = elem
                gauss_params, y_min, x_range = self.make_multigauss_fit(x, I_data, region_info)
                gauss_params = gauss_params.reshape(-1, 3)
                gauss_params[:, 0] += y_min
                total_flux = np.sum(np.sqrt(2 * np.pi) * gauss_params[:, 0] * gauss_params[:, 2])
                for local_index, gaussian in zip(indices, gauss_params):
                    amplitude, mean, stddev = gaussian
                    ar_info['Amplitude'][local_index][index] = {'freq': freq, 'amplitude': amplitude}
                    ar_info['Mean'][local_index][index] = {'freq': freq, 'mean': mean}
                    ar_info['Sigma'][local_index][index] = {'freq': freq, 'sigma': stddev}
                    ar_info['FWHM'][local_index][index] = {'freq': freq, 'fwhm': 2 * np.sqrt(2 * np.log(2)) * stddev}
                    ar_info['Range'][local_index][index] = {'freq': freq, 'x_range': (np.min(x_range), np.max(x_range))}
                    ar_info['Flux'][local_index][index] = {'freq': freq, 'flux': np.sqrt(2 * np.pi) * amplitude * stddev}
                    ar_info['Total Flux'][local_index][index] = {'freq': freq, 'flux': total_flux}
        return ar_info

    def acquire_data(self, timerange):
        scrapper = Scrapper(self.base_url, regex_pattern=self.regex_pattern, condition=self.condition, filter=self.filter)
        return scrapper.form_fileslist(timerange)

    def get_scans(self, timerange):
        file_urls = self.acquire_data(timerange)
        column_types = {
            'Date': np.dtype('U10'),
            'Time': np.dtype('U8'),
            'Azimuth': np.dtype('i2'),
            'SOLAR_R': np.dtype('float64'),
            'N_shape': np.dtype('i4'),
            'CRPIX': np.dtype('float64'),
            'CDELT1': np.dtype('float64'),
            'Pozitional Angle': np.dtype('float64'),
            'SOLAR_B': np.dtype('float64'),
            'Frequency': np.dtype('O'),
            'Flux Eficiency ': np.dtype('O'),
            'I': np.dtype('O'),
            'V': np.dtype('O'),
        }

        table = Table(names=tuple(column_types.keys()), dtype=tuple(column_types.values()))
        for file_url in file_urls:
            hdul = fits.open(file_url)
            data = hdul[0].data

            CDELT1 = hdul[0].header['CDELT1']
            CRPIX = hdul[0].header['CRPIX1']
            SOLAR_R = hdul[0].header['SOLAR_R'] * 1.01175
            SOLAR_B = hdul[0].header['SOLAR_B']
            FREQ = hdul[1].data['FREQ']
            OBS_DATE = hdul[0].header['DATE-OBS']
            OBS_TIME = hdul[0].header['TIME-OBS']
            bad_freq = np.isin(FREQ, [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062])

            AZIMUTH = hdul[0].header['AZIMUTH']
            SOL_DEC = hdul[0].header['SOL_DEC']
            SOLAR_P = hdul[0].header['SOLAR_P']
            angle = self.pozitional_angle(AZIMUTH, SOL_DEC, SOLAR_P)
            N_shape = data.shape[2]
            x = np.linspace(
                - CRPIX * CDELT1,
                (N_shape - CRPIX) * CDELT1,
                num=N_shape
            )
            flux_eficiency = self.antenna_efficiency(FREQ, SOLAR_R)
            mask, I, V = self.calibrate_QSModel(x, data, SOLAR_R, FREQ, flux_eficiency)
            I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]
            table.add_row([
                OBS_DATE.replace('/', '-'),
                OBS_TIME,
                AZIMUTH,
                SOLAR_R,
                N_shape,
                CRPIX,
                CDELT1,
                angle,
                SOLAR_B,
                FREQ,
                dict(zip(FREQ, flux_eficiency)),
                dict(zip(FREQ, I)),
                dict(zip(FREQ, V))
            ])
        return table

    def form_data(self, file_urls):
        total_table = []
        for file_url in file_urls:
            hdul = fits.open(file_url)
            data = hdul[0].data

            CDELT1 = hdul[0].header['CDELT1']
            CRPIX = hdul[0].header['CRPIX1']
            SOLAR_R = hdul[0].header['SOLAR_R'] * 1.01175
            SOLAR_B = hdul[0].header['SOLAR_B']
            FREQ = hdul[1].data['FREQ']
            OBS_DATE = hdul[0].header['DATE-OBS']
            OBS_TIME = hdul[0].header['TIME-OBS']
            bad_freq = np.isin(FREQ, [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062])

            ratan_datetime = datetime.strptime(OBS_DATE + ' ' + OBS_TIME, '%Y/%m/%d %H:%M:%S.%f')
            noaa_datetime = datetime.strptime(OBS_DATE, '%Y/%m/%d')
            diff_hours = int((ratan_datetime - noaa_datetime).total_seconds() / 3600)

            AZIMUTH = hdul[0].header['AZIMUTH']
            SOL_DEC = hdul[0].header['SOL_DEC']
            SOLAR_P = hdul[0].header['SOLAR_P']
            angle = self.pozitional_angle(AZIMUTH, SOL_DEC, SOLAR_P)

            x = np.linspace(
                - CRPIX * CDELT1,
                (data.shape[2] - CRPIX) * CDELT1,
                num=data.shape[2]
            )

            flux_eficiency = self.antenna_efficiency(FREQ, SOLAR_R)
            mask, I, V = self.calibrate_QSModel(x, data, SOLAR_R, FREQ, flux_eficiency)
            I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]

            srs = SRSClient()
            srs_table = srs.get_data(TimeRange(OBS_DATE, OBS_DATE))
            srs_table = srs_table[srs_table['ID'] == 'I']
            srs_table['Longitude'] = (
                srs_table['Longitude'] + self.differential_rotation(srs_table['Latitude']) * diff_hours / 24
            ).astype(int)

            srs_table['Latitude'], srs_table['Longitude'] = self.heliocentric_transform(
                srs_table['Latitude'],
                srs_table['Longitude'],
                SOLAR_R,
                SOLAR_B
            )

            srs_table['Latitude'], srs_table['Longitude'] = self.pozitional_rotation(
                srs_table['Latitude'],
                srs_table['Longitude'],
                angle
            )

            ar_info = self.active_regions_search(srs_table, x, V, mask)
            ar_amount = len(ar_info)
            ar_info.add_column(Column(name='Azimuth', data=[AZIMUTH]*ar_amount, dtype=('i2')), index=1)
            ar_info.add_column(Column(name='Amplitude', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(Column(name='Mean', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(Column(name='Sigma', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(Column(name='FWHM', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(Column(name='Flux', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(Column(name='Total Flux', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(Column(name='Range', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))

            ratan_data = list(zip(FREQ, I, V))
            gauss_analysis_info = self.gauss_analysis(x, ratan_data, ar_info)
            total_table.append(gauss_analysis_info)
        return Table(vstack(total_table))

    def get_data(self, timerange):
        file_urls = self.acquire_data(timerange)
        return self.form_data(file_urls)

