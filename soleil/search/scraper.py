import re
from urllib.parse import urlsplit
from ftplib import FTP
import bs4 as BeautifulSoup
import numpy as np
from functools import lru_cache

from astropy.time import Time
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

from soleil.time import TimeRange

TIME_REGEX = {'%Y': r'\d{4}', '%y': r'\d{2}',
              '%b': '[A-Z][a-z]{2}', '%m': r'\d{2}',
              '%d': r'\d{2}', '%j': r'\d{3}',
              '%H': r'\d{2}',
              '%M': r'\d{2}',
              '%S': r'\d{2}'}

class Scrapper: 
    def __init__(self, baseurl, http_parse=None, filter_func=None, regex=True, **kwargs):
        # if 'pattern' is constant i can pass it with the regex symbols 
        # if not - i can format it using some passed conditions and arguments
        # 'filter' - helps to filter future files list according to some 
        ###  conditions (for example, "bad" observations)
        if regex: 
            self.baseurl = baseurl
        else: 
            self.baseurl = baseurl.format(**kwargs) 

        self.domain = f"{urlsplit(self.baseurl).scheme}://{urlsplit(self.baseurl).netloc}/"
        self.filter = filter_func

    
    def smallest_significant_pattern(self, pattern):
        """
        Some of them are here: https://fits.gsfc.nasa.gov/iso-time.html
        """
        try:
            if any(second in pattern for second in ['%S']):
                return relativedelta(seconds=1)
            elif any(minute in pattern for minute in ['%M']):
                return relativedelta(minutes=1)
            elif any(hour in pattern for hour in ['%H']):
                return relativedelta(hours=1)
            elif any(day in pattern for day in ['%d', '%j']):
                return relativedelta(days=1)
            elif any(month in pattern for month in ['%m', '%b']):
                return relativedelta(months=1)
            if any(year in pattern for year in ['%y', '%Y']):
                return relativedelta(years=1)
            else: 
                return None
        except Exception:
            raise


    @staticmethod
    def floor_datetime(date, timestep):
        date = date.to_datetime()
        if timestep.years > 0:
            return date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif timestep.months > 0:
            return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif timestep.days > 0:
            return date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timestep.hours > 0:
            return date.replace(minute=0, second=0, microsecond=0)
        elif timestep.minutes > 0:
            return date.replace(second=0, microsecond=0)
        return date


    def range(self, timerange: TimeRange):
        filepath_pattern = '/'.join(self.baseurl.split('/')[:-1]) + '/'
        smallest_step = self.smallest_significant_pattern(filepath_pattern)
        if smallest_step is None: 
            return [filepath_pattern]
        directories = []
        current_date = self.floor_datetime(timerange.start, smallest_step)
        end_date = self.floor_datetime(timerange.end, smallest_step) + smallest_step
        while current_date < end_date: 
            directories.append(current_date.strftime(filepath_pattern))
            current_date += smallest_step
        return directories
    

    def extract_date_from_url(self, url):
        def url2list(text):
            return re.sub(r'\.|_', '/', text).split('/')
        pattern_parts = url2list(self.baseurl)
        url_parts = url2list(url)
        time_formats = ['%Y', '%y', '%b', '%B', '%m', '%d', '%j',
                        '%H', '%M', '%S']
        extracted_date, extracted_pattern = [], []
        for p_elem, u_elem in zip(pattern_parts, url_parts):
            present_formats = [x for x in time_formats if x in p_elem]
            part_to_remove = re.split('%.', p_elem)

            for candidate_to_remove in part_to_remove:
                if candidate_to_remove:
                    u_elem = u_elem.replace(candidate_to_remove, '', 1)
                    p_elem = p_elem.replace(candidate_to_remove, '', 1)

            extracted_date.append(u_elem)
            extracted_pattern.append(p_elem)
            time_formats = [fmt for fmt in time_formats if fmt not in present_formats]

        # Find the index of the fullest pattern
        fullest_pattern_index = extracted_pattern.index(max(extracted_pattern, key=len))
        # Find the corresponding date part
        date_part = extracted_date[fullest_pattern_index]
        return Time.strptime(date_part, extracted_pattern[fullest_pattern_index])
  

    def valid_date_from_url(self, url):
        pattern = self.baseurl
        # Replace datetime formats in the pattern string with their corresponding regex patterns
        for time_format, regex in TIME_REGEX.items():
            pattern = pattern.replace(time_format, regex)
        # Create a regex pattern object
        pattern_obj = re.compile(pattern)
        # Check if the URL matches the pattern
        return pattern_obj.fullmatch(url) is not None


    def check_date_in_timerange(self, url, timerange):
        file_date = self.extract_date_from_url(url).to_datetime()
        smallest_pattern = self.smallest_significant_pattern(self.baseurl)
        file_range = TimeRange(file_date, file_date + smallest_pattern)
        return timerange.have_intersection(file_range)


    def ftpfiles(self, timerange):
        directories = self.range(timerange)
        file_urls = []
        ftpurl = urlsplit(directories[0]).netloc
        with FTP(ftpurl, user="anonymous", passwd="soleil@package") as ftp:
            for current_directory in directories:
                try: 
                    ftp.cwd(urlsplit(current_directory).path)
                except Exception as e:
                    print(f'FTP CWD tried: {e}')
                    continue
                for file_name in ftp.nlst():
                    file_path = current_directory + file_name
                    if self.valid_date_from_url(file_path):
                        if self.check_date_in_timerange(file_path, timerange): 
                            file_urls.append(file_path)
        return file_urls


    def httpfiles(self, timerange):
        pass


    def form_fileslist(self, timerange): 
        # NOAA, for example
        if urlsplit(self.baseurl).scheme == 'ftp':
            return self.ftpfiles(timerange)
        # RATAN, for example
        if urlsplit(self.baseurl).scheme == 'http':
            return self.httpfiles(timerange)
        
