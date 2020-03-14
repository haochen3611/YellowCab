import pandas as pd
from typing import Union
import numpy as np
import os
import re
import time


FILE = 'data/raw/yellow_tripdata_2019-12.csv'
ZONE = 'data/taxi+_zone_lookup.csv'


class DataProcessor:
    DTYPE = []

    def __init__(self, data: Union[str, pd.DataFrame], **kwargs):
        assert isinstance(data, (str, pd.DataFrame)), f'invalid file type: {type(data)}, need \'str\' or \'dataframe\''
        if isinstance(data, str):
            data = pd.read_csv(data, low_memory=False, index_col=False)
            assert isinstance(data, pd.DataFrame)
        self._data = data
        self._raw = data.copy()

        try:
            self._loc_zone_table = kwargs.pop('loc_zone')
        except KeyError:
            pass
        else:
            self._process_zone_table()

        self._simple_process()

    @property
    def data(self):
        return self._data.reset_index(drop=True)

    @property
    def raw(self):
        return self._raw

    def reset(self):
        self._data = self._raw.copy()

    def _return(self, *args):
        assert len(args) > 0, 'expect at least 1 position argument'
        if len(args) == 1:
            if hasattr(self, '_loc_zone_table'):
                return DataProcessor(data=args[0], loc_zone=self._loc_zone_table)
            else:
                return DataProcessor(data=args[0])
        else:
            if hasattr(self, '_loc_zone_table'):
                return [DataProcessor(data=ag, loc_zone=self._loc_zone_table) for ag in args]
            else:
                return [DataProcessor(data=ag) for ag in args]

    def _simple_process(self):
        self._data.loc[:, 'tpep_pickup_datetime'] = pd.to_datetime(self.data.loc[:, 'tpep_pickup_datetime'])
        self._data.loc[:, 'tpep_dropoff_datetime'] = pd.to_datetime((self.data.loc[:, 'tpep_dropoff_datetime']))
        self._data.loc[:, 'trip_time'] = (self._data.tpep_dropoff_datetime - self._data.tpep_pickup_datetime)\
            .apply(lambda x: x.total_seconds())
        self._data = self._data.loc[(self._data.trip_time > 60) & (self._data.trip_time < 7200)]
        self._data = self._data.loc[(self._data.trip_distance > 0.1) & (self._data.trip_distance < 20)]

    def _process_zone_table(self):
        assert isinstance(self._loc_zone_table, (str, pd.DataFrame, dict)), \
            f'invalid file type: {type(self._loc_zone_table)}, need \'str\', \'dataframe\', or \'dict\''
        if isinstance(self._loc_zone_table, str):
            self._loc_zone_table = pd.read_csv(self._loc_zone_table, low_memory=False, index_col=False)
            assert isinstance(self._loc_zone_table, pd.DataFrame)
        if isinstance(self._loc_zone_table, pd.DataFrame):
            for c in ['LocationID', 'Borough']:
                assert c in self._loc_zone_table.columns, f'{c} is not in the table'
            groups = self._loc_zone_table.groupby(by='Borough').groups
            for key in groups:
                groups[key] = self._loc_zone_table.loc[groups[key], 'LocationID']
            self._loc_zone_table = groups
        assert isinstance(self._loc_zone_table, dict)

    def _filter_location(self, location, column, **kwargs):
        assert column in self._data.columns, f'column not found: {column}'
        assert isinstance(location, (str, int)), \
            f'invalid \'location\' type: {type(location)}, need \'str\' or \'int\''

        if isinstance(location, str):
            try:
                loc_zone = kwargs.pop('loc_zone')
            except KeyError:
                if not hasattr(self, '_loc_zone_table'):
                    raise TypeError('Missing location zone file. Add it by using \'loc_zone\' argument.')
            else:
                setattr(self, '_loc_zone_table', loc_zone)
                self._process_zone_table()

            assert location in self._loc_zone_table.keys(), \
                f'Unknown location: {location}, must be in {self._loc_zone_table.keys()}'
            filtered = self._data.loc[self._data['PULocationID'].isin(self._loc_zone_table[location])]
        else:
            if self._data.loc['PULocationID'].dtype == 'int64':
                filtered = self._data.loc[self._data['PULocationID'] == location]
            else:
                filtered = self._data.loc[self._data['PULocationID'].astype('int64') == location]
        return filtered

    def filter_pickup_location(self, location: Union[str, int], inplace: bool = True, **kwargs):
        filtered = self._filter_location(location, 'PULocationID', **kwargs)
        if not inplace:
            return self._return(filtered)
        self._data = filtered

    def filter_dropoff_location(self, location: Union[str, int], inplace: bool = True, **kwargs):
        filtered = self._filter_location(location, 'DOLocationID', **kwargs)
        if not inplace:
            return self._return(filtered)
        self._data = filtered

    def filter_pickup_time(self, start: int, end: int, inplace: bool = True):
        assert isinstance(start, int), f'invalid \'start\' type: {type(start)}'
        assert isinstance(end, int), f'invalid \'end\' type: {type(end)}'
        assert 0 <= start < end <= 24, f'invalid start={start} and end={end}'

        hour_lst = self._data.loc[:, 'tpep_pickup_datetime'].apply(func=lambda x: x.hour)
        filtered = self._data.loc[(hour_lst >= start) & (hour_lst < end)]
        if not inplace:
            return self._return(filtered)
        self._data = filtered

    def filter_weekday(self, weekend: bool = False, inplace: bool = True):
        assert isinstance(weekend, bool), f'invalid \'weekend\' type: {type(weekend)}'

        day_lst = self._data.loc[:, 'tpep_pickup_datetime'].apply(func=lambda x: x.weekday())
        if weekend:
            filtered = self._data.loc[day_lst >= 5]
        else:
            filtered = self._data.loc[day_lst < 5]
        if not inplace:
            return self._return(filtered)
        self._data = filtered

    def sort_by(self, by: str, ascending: bool = True, inplace: bool = True):
        assert isinstance(by, str), f'invalid \'by\' type: {type(by)}'
        sorted_ = self._data.sort_values(by=by, ascending=ascending)
        if not inplace:
            return self._return(sorted_)
        self._data = sorted_


def get_average_arrival_time(data: pd.DataFrame, pickup: int, dropoff: int):

    pickup = data.loc[:, 'PULocationID'] == pickup
    dropoff = data.loc[:, 'DOLocationID'] == dropoff
    selected = data.loc[pickup & dropoff]
    num_days = len(selected.tpep_pickup_datetime.unique())
    num_trips = len(selected.index)
    if num_days > num_trips or num_trips < 1:
        return np.nan
    return 3600 * num_days / num_trips


def get_interarrival_time(data: pd.DataFrame, aam: float, pickup: int, dropoff: int):
    pickup = data.loc[:, 'PULocationID'] == pickup
    dropoff = data.loc[:, 'DOLocationID'] == dropoff
    selected = data.loc[pickup & dropoff]
    trip_time = selected.trip_time

    if np.isnan(aam) or len(trip_time) < 1:
        return np.nan
    return trip_time.mean()


def data_process_routine(data_file, zone_file):
    try:
        year, month = parse_date_from_filename(data_file)
    except ValueError:
        year, month = None, None
    data_path = os.path.join(RAW_DIR, data_file)
    zone_path = os.path.join(DATA_DIR, zone_file)
    dp = DataProcessor(data=data_path, loc_zone=zone_path)
    dp.filter_pickup_location('Manhattan')
    dp.filter_dropoff_location('Manhattan')
    dp.filter_pickup_time(start=8, end=9)
    dp.filter_weekday()

    dat = dp.data
    pu_lst = dat.PULocationID.unique()
    do_lst = dat.DOLocationID.unique()
    aam = pd.DataFrame(index=pu_lst, columns=do_lst)
    iat = pd.DataFrame(index=pu_lst, columns=do_lst)
    for pu in pu_lst:
        for do in do_lst:
            aam.loc[pu, do] = get_average_arrival_time(dat, pu, do)
            iat.loc[pu, do] = get_interarrival_time(dat, aam.loc[pu, do], pu, do)

    if year is not None and month is not None:
        aam.to_csv(os.path.join(AAM_DIR, f'aam-{year}-{month}.csv'), na_rep='NA', line_terminator='\n')
        iat.to_csv(os.path.join(ATM_DIR, f'atm-{year}-{month}.csv'), na_rep='NA', line_terminator='\n')
        lock.acquire()
        print(f'{year}-{month}-done!')
        lock.release()
    else:
        aam.to_csv(os.path.join(AAM_DIR, f'aam-{data_file}.csv'), na_rep='NA', line_terminator='\n')
        iat.to_csv(os.path.join(ATM_DIR, f'atm-{data_file}.csv'), na_rep='NA', line_terminator='\n')
        lock.acquire()
        print(f'{data_file}-done!')
        lock.release()


def parse_date_from_filename(fname):
    date_par = re.compile(r'\d\d\d\d-\d\d')
    se = date_par.search(fname)
    if se is not None:
        seg = se.string.split('-')
        year = seg[0]
        month = seg[1]
    else:
        raise ValueError('No date in file name')
    return year, month


def init(lk):
    global lock
    lock = lk


if __name__ == '__main__':
    from download_data import download_file_parallel, RAW_DIR, DATA_DIR, AAM_DIR, ATM_DIR
    import multiprocessing as mp

    lk_ = mp.Lock()
    data_files, zone_file_ = download_file_parallel(4)
    items = [(df, zone_file_) for df in data_files]
    with mp.Pool(mp.cpu_count(),
                 initializer=init, initargs=(lk_, )) as pool:
        pool.starmap(data_process_routine, items)
