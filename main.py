import pandas as pd
from typing import Union


FILE = 'data/yellow_tripdata_2019-12.csv'


class DataProcessor:
    DTYPE = []

    def __init__(self, data: Union[str, pd.DataFrame], **kwargs):
        assert isinstance(data, (str, pd.DataFrame)), f'invalid file type: {type(data)}, need \'str\' or \'dataframe\''
        if isinstance(data, str):
            data = pd.read_csv(data, low_memory=False, index_col=False)
        self._data = data

        try:
            self._loc_zone_table = kwargs.pop('loc_zone')
        except KeyError:
            pass
        else:
            assert isinstance(data, (str, pd.DataFrame)), \
                f'invalid file type: {type(data)}, need \'str\' or \'dataframe\''
            if isinstance(data, str):
                self._loc_zone_table = pd.read_csv(self._loc_zone_table, low_memory=False, index_col=False)
            for c in ['LocationID', 'Borough']:
                assert c in self._loc_zone_table.columns, f'{c} is not in the table'

        self._simple_process()

    @property
    def data(self):
        return self._data

    def _simple_process(self):
        self._data.loc[:, 'tpep_pickup_datetime'] = pd.to_datetime(self.data.loc[:, 'tpep_pickup_datetime'])
        self._data.loc[:, 'tpep_dropoff_datetime'] = pd.to_datetime((self.data.loc[:, 'tpep_dropoff_datetime']))
        groups = self._loc_zone_table.groupby(by='Borough').groups
        for key in groups:
            groups[key] = self._loc_zone_table.loc[groups[key], 'LocationID']
        self._loc_zone_table = groups

    def filter_pickup_location(self, location: Union[str, int], **kwargs):
        assert isinstance(location, (str, int)), \
            f'invalid \'location\' type: {type(location)}, need \'str\' or \'int\''

        if isinstance(location, str):
            try:
                loc_zone = kwargs.pop('loc_zone')
            except KeyError:
                if hasattr(self, '_loc_zone_table'):
                    loc_zone = self._loc_zone_table
                else:
                    raise TypeError('Missing location zone file. Add it by using \'loc_zone\' argument.')
            assert location in self._loc_zone_table.keys(), \
                f'Unknown location: {location}, must be in {self._loc_zone_table.keys()}'
            return self._data.loc[self._data['PULocationID'].isin(self._loc_zone_table[location])]
        else:
            if self._data.loc['PULocationID'].dtype == 'int64':
                return self._data.loc[self._data['PULocationID'] == location]
            else:
                return self._data.loc[self._data['PULocationID'].astype('int64') == location]

    def filter_pickup_time(self, start: int, end: int):
        assert isinstance(start, int), f'invalid \'start\' type: {type(start)}'
        assert isinstance(end, int), f'invalid \'end\' type: {type(end)}'
        assert 0 <= start < end <= 12, f'invalid start={start} and end={end}'

        hour_lst = self._data.loc[:, 'tpep_pickup_datetime'].apply(func=lambda x: x.hour)
        return self._data.loc[(hour_lst >= start) & (hour_lst < end)]

    def filter_weekday(self, weekend: bool = False):
        assert isinstance(weekend, bool), f'invalid \'weekend\' type: {type(weekend)}'

        day_lst = self._data.loc[:, 'tpep_pickup_datetime'].apply(func=lambda x: x.weekday())
        if weekend:
            return self._data.loc[day_lst >= 5]
        return self._data.loc[day_lst < 5]

    def sort_by(self, by: str, ascending: bool = True):
        assert isinstance(by, str), f'invalid \'by\' type: {type(by)}'
        return self._data.sort_values(by=by, ascending=ascending)








