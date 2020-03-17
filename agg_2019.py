from main import DataProcessor
from util import download_file_parallel, LOG_DIR, \
        DATA_DIR, AAM_DIR, ATM_DIR, set_destination, parse_date_from_filename, \
        get_csv_file_from_dir, filter_csv_file_by_time, handle_parser_error, \
        BadLineError, ColumnNotFoundError
import os
import time
import sys
import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse as ap


LOG = os.path.join(LOG_DIR, f'process-{int(time.time())}.log')


def get_ndays_ntrips(data: pd.DataFrame, pickup: int, dropoff: int):
    pickup = data.loc[:, 'PULocationID'] == pickup
    dropoff = data.loc[:, 'DOLocationID'] == dropoff
    selected = data.loc[pickup & dropoff]
    day_lst = selected.tpep_pickup_datetime.apply(lambda x: x.day)
    num_days = len(day_lst.unique())
    num_trips = len(selected.index)
    return num_days, num_trips


def get_interarrival_time(data: pd.DataFrame, pickup: int, dropoff: int):
    pickup = data.loc[:, 'PULocationID'] == pickup
    dropoff = data.loc[:, 'DOLocationID'] == dropoff
    selected = data.loc[pickup & dropoff]
    trip_time = selected.trip_time

    if len(trip_time) < 1:
        return 0
    return trip_time.mean()


def aggregate_year_data(data_file, zone_file, pu_lst, weekday=True, start_time=0, location='Manhattan'):
    try:
        year, month = parse_date_from_filename(data_file)
        name = f'{year}-{month}'
    except ValueError:
        name = data_file.split('/')[-1].split('.')[0]

    wkd = 'wd' if weekday else 'wn'
    data_path = os.path.join(RAW_DIR, data_file)
    zone_path = os.path.join(DATA_DIR, zone_file)

    try:
        dp = DataProcessor(data=data_path, loc_zone=zone_path)
        dp.filter_pickup_time(start=start_time, end=start_time+1)
        dp.filter_pickup_location(location)
        dp.filter_dropoff_location(location)
        dp.filter_weekday(weekend=not weekday)
    except Exception as err:
        with open(LOG, 'a') as ff:
            ff.write(f'{name}-{wkd}-{start_time}' + f'...{sys.exc_info()[0]}: {err}\n')
        print(f'{name}-{wkd}-{start_time}' + f'...{sys.exc_info()[0]}: {err}')
        return 0, 0, 0
    else:
        dat = dp.data
        n_days = pd.DataFrame(index=pu_lst, columns=pu_lst)
        n_trips = pd.DataFrame(index=pu_lst, columns=pu_lst)
        atm = pd.DataFrame(index=pu_lst, columns=pu_lst)

        for pu in pu_lst:
            for do in pu_lst:
                d, t = get_ndays_ntrips(dat, pu, do)
                n_days.loc[pu, do] += d
                n_trips.loc[pu, do] += t
                atm.loc[pu, do] = get_interarrival_time(dat, pu, do)
        return n_days, n_trips, atm


def combine_results(res: list, pu_lst):
    for w, wd_ in enumerate(['wd', 'wn']):
        for hr_ in range(24):
            ds = pd.DataFrame(index=pu_lst, columns=pu_lst, data=0)
            ts = pd.DataFrame(index=pu_lst, columns=pu_lst, data=0)
            atm = pd.DataFrame(index=pu_lst, columns=pu_lst, data=0)
            for f_, ff in enumerate(files_19):
                idx = f_ + hr_*len(files_19) + w*len(files_19)*24
                d, t, a = res[idx]
                ds += d
                ts += t
                atm += a
            aam = 3600.*(ds.div(ts).fillna(0))
            atm = atm.div(len(files_19))
            aam.to_csv(os.path.join(AAM_DIR, f'aam-2019-{wd_}-{hr_}.csv'),
                       na_rep='NA', line_terminator='\n')
            atm.to_csv(os.path.join(AAM_DIR, f'atm-2019-{wd_}-{hr_}.csv'),
                       na_rep='NA', line_terminator='\n')


if __name__ == '__main__':

    par = ap.ArgumentParser(prog='data processor', description='CLI input to data processor')
    par.add_argument('--dest', nargs='?', metavar='<RAW DATA DIR>', type=str, default=None)
    arg = par.parse_args()

    dest = arg.dest
    RAW_DIR = set_destination(dest)

    files_19 = get_csv_file_from_dir(RAW_DIR)
    zone_file_ = 'taxi+_zone_lookup.csv'
    zone_table = pd.read_csv(os.path.join(DATA_DIR, zone_file_), low_memory=False, index_col=False)
    zone_gp = zone_table.groupby(by='Borough').groups
    for key in zone_gp:
        zone_gp[key] = zone_table.loc[zone_gp[key], 'LocationID']

    man_id = sorted(zone_gp['Manhattan'].values)

    item = []
    for wd in [True, False]:
        for hr in range(24):
            for f in files_19:
                item.append((f, zone_file_, man_id, wd, hr))

    with mp.Pool(20) as pool:
        results = pool.starmap(aggregate_year_data, item)

    combine_results(results, man_id)
