from datetime import timedelta

import astropy.units as u
from astropy.time import Time, TimeDelta

from soleil.time.time import parse_time, check_equal_time

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

class TimeRange:
    def __init__(self, a, b=None, format=None):
        self._start_time = None 
        self._end_time = None

        # if TimeRange passed 
        if isinstance(a, TimeRange):
            self.__dict__ = a.__dict__.copy()
            return  
        
        # if b is None and a is array-like or tuple.
        # for example, data and delta to the end of timerange
        if b is None: 
            x = parse_time(a[0], format=format)
            if len(a) != 2:
                raise ValueError('"a" must have two elements')
            else: 
                y = a[1]
        else: 
            x = parse_time(a, format=format)
            y = b 

        # if y is timedelta
        if isinstance(y, timedelta):
            y = TimeDelta(y, format='datetime')

        # create TimeRange in case of (start_date, delta)
        if isinstance(y, TimeDelta):
            # positive delta 
            if y.jd >= 0:
                self._start_time = x 
                self._end_time = x + y
            else: 
                self._start_time = x + y
                self._end_time = x
            return

        # otherwise, b is something date-like
        y = parse_time(y, format=format)
        if isinstance(y, Time):
            if x < y:
                self._start_time = x 
                self._end_time = y
            else: 
                self._start_time = y 
                self._end_time = x


    @property
    def start(self):
        return self._start_time


    @property
    def end(self):
        return self._end_time
    

    @property
    def delta(self):
        return self._end_time - self._start_time

    @property
    def days(self):
        return self.delta.to('day').value
    
    
    @property
    def hours(self):
        return self.delta.to('hour').value
    

    @property
    def minutes(self):
        return self.delta.to('minute').value
    

    @property
    def seconds(self):
        return self.delta.to('second').value
    

    def __eq__(self, other): 
        if isinstance(other, TimeRange):
            return check_equal_time(self.start, other.start) and check_equal_time(self.end, other.end)
        return NotImplemented
    

    def __ne__(self, other): 
        if isinstance(other, TimeRange):
            return not (check_equal_time(self.start, other.start) and check_equal_time(self.end, other.end))
        return NotImplemented
    

    def __contains__(self, time):
        time_to_check = parse_time(time)
        return time_to_check >= self.start and time_to_check <= self.end
    

    def __repr__(self):
        start_time = self.start.strftime(TIME_FORMAT)
        end_time = self.end.strftime(TIME_FORMAT)
        full_name = f'{self.__class__.__module__}.{self.__class__.__name__}'
        return (
            f'<{full_name} object at {hex(id(self))}>' + 
            '\nStart:'.ljust(12) + start_time + 
            '\nEnd:'.ljust(12) + end_time + 
            '\nDuration:'.ljust(12) + f'{str(self.days)} days | {str(self.hours)} hours | {str(self.minutes)} minutes | {str(self.seconds)} seconds'
        )
    

    def __str__(self):
        start_time = self.start.strftime(TIME_FORMAT)
        end_time = self.end.strftime(TIME_FORMAT)
        return (
            f'({start_time}, {end_time})'
        )
    

    def have_intersection(self, other):
        intersection_lower = max(self.start, other.start)
        intersection_upper = min(self.end, other.end)
        return intersection_lower <= intersection_upper


    def get_dates(self, filter=None): 
        delta = self.end.to_datetime().date() - self.start.to_datetime().date()
        t_format = "%Y-%m-%d"
        dates_list = [parse_time(self.start.strftime(t_format)) + TimeDelta(i * u.day)
                      for i in range(delta.days + 1)]
        # filter is a list of dates to be excluded, maybe should add typings
        if filter: 
            dates_list = [date for date in dates_list if date not in parse_time(filter)]
        return dates_list
    

    def moving_window(self, window_size, window_period):
        if not isinstance(window_size, TimeDelta):
            window_size = TimeDelta(window_size)
        if not isinstance(window_period, TimeDelta):
            window_period = TimeDelta(window_period)


        window_number = 1
        times = [TimeRange(self.start, self.start + window_size)]

        while times[-1].end < self.end:
            times.append(
                TimeRange(
                    self.start + window_number * window_period, 
                    self.start + window_number * window_period + window_size, 
                )
            )
            print(times)
            window_number += 1
        return times
        

    def equal_split(self, n_splits=2):
        if n_splits <= 0:
            raise ValueError('n must be greater or equal than 1')
        subranges = []
        prev_time = self.start
        next_time = None 
        for _ in range(n_splits):
            next_time = prev_time + self.delta / n_splits 
            next_range = TimeRange(prev_time, next_time)
            subranges.append(next_range)
            prev_time = next_time
        return subranges    


    def shift_forward(self, delta: TimeDelta = None): 
        delta = delta if delta else self.delta 
        self._start_time += delta
        self._end_time += delta
        return self
    

    def shift_backward(self, delta: TimeDelta = None): 
        delta = delta if delta else self.delta
        self._start_time -= delta
        self._end_time -= delta
        return self
    

    def extend_range(self, start_delta: TimeDelta, end_delta: TimeDelta): 
        self._start_time += start_delta
        self._end_time += end_delta
        return self
