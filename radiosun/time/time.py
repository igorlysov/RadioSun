import re
from datetime import date, datetime
from functools import singledispatch
import numpy as np
import pandas 
import astropy.time
import astropy.units as u
from astropy.time import Time


TIME_REGEX = {
    '%Y': r'(?P<year>\d{4})',
    '%j': r'(?P<dayofyear>\d{3})',
    '%m': r'(?P<month>\d{1,2})',
    '%d': r'(?P<day>\d{1,2})',
    '%H': r'(?P<hour>\d{1,2})',
    '%M': r'(?P<minute>\d{1,2})',
    '%S': r'(?P<second>\d{1,2})',
    '%f': r'(?P<microsecond>\d+)',
    '%b': r'(?P<month_str>[a-zA-Z]+)',
}


COMMON_TIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",  # 2007-05-04T21:08:12.999999
    "%Y/%m/%dT%H:%M:%S.%f",  # 2007/05/04T21:08:12.999999
    "%Y-%m-%dT%H:%M:%S.%fZ",  # 2007-05-04T21:08:12.999Z
    "%Y-%m-%dT%H:%M:%S",  # 2007-05-04T21:08:12
    "%Y/%m/%dT%H:%M:%S",  # 2007/05/04T21:08:12
    "%Y%m%dT%H%M%S.%f",  # 20070504T210812.999999
    "%Y%m%dT%H%M%S",  # 20070504T210812
    "%Y/%m/%d %H:%M:%S",  # 2007/05/04 21:08:12
    "%Y/%m/%d %H:%M",  # 2007/05/04 21:08
    "%Y/%m/%d %H:%M:%S.%f",  # 2007/05/04 21:08:12.999999
    "%Y-%m-%d %H:%M:%S.%f",  # 2007-05-04 21:08:12.999999
    "%Y-%m-%d %H:%M:%S",  # 2007-05-04 21:08:12
    "%Y-%m-%d %H:%M",  # 2007-05-04 21:08
    "%Y-%b-%d %H:%M:%S.%f",  # 2007-May-04 21:08:12.999999
    "%Y-%b-%d %H:%M:%S",  # 2007-May-04 21:08:12
    "%Y-%b-%d %H:%M",  # 2007-May-04 21:08
    "%Y-%b-%d",  # 2007-May-04
    "%Y-%m-%d",  # 2007-05-04
    "%Y/%m/%d",  # 2007/05/04
    "%d-%b-%Y",  # 04-May-2007
    "%d-%b-%Y %H:%M:%S",  # 04-May-2007 21:08:12
    "%d-%b-%Y %H:%M:%S.%f",  # 04-May-2007 21:08:12.999999
    "%Y%m%d_%H%M%S",  # 20070504_210812
    "%Y:%j:%H:%M:%S",  # 2012:124:21:08:12
    "%Y:%j:%H:%M:%S.%f",  # 2012:124:21:08:12.999999
    "%Y%m%d%H%M%S",  # 20140101000001 (JSOC/VSO Export/Downloads)
    "%Y.%m.%d_%H:%M:%S_TAI",  # 2016.05.04_21:08:12_TAI - JSOC
    "%Y.%m.%d_%H:%M:%S_UTC",  # 2016.05.04_21:08:12_UTC - JSOC
    "%Y.%m.%d_%H:%M:%S",  # 2016.05.04_21:08:12 - JSOC
    "%Y/%m/%dT%H:%M",  # 2007/05/04T21:08
]

def check_equal_time(t1, t2):
    if abs(t2 - t1) < 1 * u.nanosecond:
        return True 
    return False


def get_time_format(time_string):
    for time_format in COMMON_TIME_FORMATS:
        if regex_time(time_string, time_format) is not None:
            return time_format


def regex_time(time_string, format):
    for key, value in TIME_REGEX.items():
        format = format.replace(key, value)
    match = re.match(format, time_string)
    if match is None:
        return None
    return time_string


@singledispatch
def create_time(time_string, format=None, **kwargs):
    return Time(time_string, format=format, **kwargs)


@create_time.register(pandas.Timestamp)
def create_time_pandasTimestamp(time_string, **kwargs):
    return Time(time_string.asm8)


@create_time.register(pandas.Series)
def create_time_pandasSeries(time_string, **kwargs):
    return Time(time_string.tolist(), **kwargs)


@create_time.register(pandas.DatetimeIndex)
def create_time_pandasDatetimeIndex(time_string, **kwargs):
    return Time(time_string.tolist(), **kwargs)


@create_time.register(datetime)
def create_time_datetime(time_string, **kwargs):
    return Time(time_string, **kwargs)


@create_time.register(date)
def create_time_date(time_string, **kwargs):
    return Time(time_string.isoformat(), **kwargs)


@create_time.register(np.datetime64)
def create_time_npdatetime64(time_string, **kwargs):
    return Time(str(time_string.astype('M8[ns]')), **kwargs)


@create_time.register(np.ndarray)
def create_time_npndarray(time_string, **kwargs):
    if 'datetime64' in str(time_string.dtype):
        return Time([str(dt.astype('M8[ns]')) for dt in time_string], **kwargs)
    else:
        return create_time.dispatch(object)(time_string, **kwargs)


@create_time.register(astropy.time.Time)
def create_time_astropy(time_string, **kwargs):
    return time_string


@create_time.register(list)
def create_time_list(time_list, format=None, **kwargs):
    item = time_list[0]
    if isinstance(item, str) and format is None:
        time_format = get_time_format(item)
        return Time.strptime(time_list, time_format, **kwargs)

    return create_time_list.dispatch(object)(time_list, format, **kwargs)


@create_time.register(str)
def create_time_str(time_string, **kwargs):
    for time_format in COMMON_TIME_FORMATS:
        try:
            try:
                ts = regex_time(time_string, time_format)
            except TypeError:
                break
            if ts is None:
                continue
            t = Time.strptime(ts, time_format, **kwargs)
            return t
        except ValueError:
            pass

    return create_time.dispatch(object)(time_string, **kwargs)


def parse_time(time_string, format=None, **kwargs):
    if isinstance(time_string, str) and time_string == 'now':
        rt = Time.now()
    else:
        rt = create_time(time_string, format=format, **kwargs)
    return rt
