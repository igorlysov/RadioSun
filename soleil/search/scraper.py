import re
from urllib.parse import urlsplit
from ftplib import FTP
import bs4 as BeautifulSoup
import numpy as np
from functools import lru_cache
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

from soleil.time import TimeRange

class Scrapper: 
    def __init__(self, baseurl, filter_func=None, regex=True, **kwargs):
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
        if '/' in self.baseurl:
            filepath_pattern = '/'.join(self.baseurl.split('/')[:-1]) + '/'
        smallest_step = self.smallest_significant_pattern(filepath_pattern)
        print(smallest_step)
        if smallest_step is None: 
            return [filepath_pattern]
        directories = []
        current_date = self.floor_datetime(timerange.start, smallest_step)
        end_date = self.floor_datetime(timerange.end, smallest_step) + smallest_step
        print(current_date)
        print(end_date)
        while current_date < end_date: 
            directories.append(current_date.strftime(filepath_pattern))
            current_date += smallest_step
        return directories


    def ftpfiles(self, timerange, filter=None):
        directories = self.range(timerange)
        filesurls = []
        ftpurl = urlsplit(directories[0]).netloc
        with FTP(ftpurl, user="anonymous", passwd="soleil@package") as ftp:
            pass


    def httpfiles(self, timerange, filter=None):
        pass 


    def fileslist(self, timerange): 
        # NOAA, for example
        if urlsplit(self.baseurl).scheme == 'ftp':
            return self.ftpfiles(timerange)
        # RATAN, for example
        if urlsplit(self.baseurl).scheme == 'http':
            return self.httpfiles(timerange)
        

    def extract_data(self, timerange):
        pass


base_url = r'ftp://ftp.ngdc.noaa.gov/STP/swpc_products/daily_reports/solar_region_summaries/%Y/%m/%Y%m%dSRS.txt'
##pattern = '{}/{year:4d}/{month:2d}/{year:4d}{month:2d}{day:2d}SRS.txt'
scraper = Scrapper(base_url)
##
##print(f"{urlsplit(base_url).scheme}://{urlsplit(base_url).netloc}/")
#
g = TimeRange('2023-10-10 01:10:12', '2023-11-12 01:10:12')
#
#t = '/'.join(base_url.split('/')[:-1]) + '/'
print(scraper.domain)


    


