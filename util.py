from bs4 import BeautifulSoup
import requests
import re
import multiprocessing as mp
import time
import os
import sys
import glob
from itertools import islice
from typing import TextIO


URL = "https://nyc-tlc.s3.amazonaws.com/"
CWD = os.path.dirname(os.path.realpath(__file__))

DEST = CWD
RAW_DIR = os.path.join(DEST, 'data/raw')
DATA_DIR = os.path.join(DEST, 'data')
LOG_DIR = os.path.join(DEST, 'logs')
AAM_DIR = os.path.join(DEST, 'data/aam')
ATM_DIR = os.path.join(DEST, 'data/atm')
if not os.path.exists(RAW_DIR):
    os.makedirs(RAW_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists(AAM_DIR):
    os.makedirs(AAM_DIR)

if not os.path.exists(ATM_DIR):
    os.makedirs(ATM_DIR)

LOG = os.path.join(LOG_DIR, f'download-{int(time.time())}.log')


def parse_date_from_filename(fname):
    date_par = re.compile(r'(?P<year>\d\d\d\d)-(?P<month>\d\d)')
    se = date_par.search(fname)
    if se is not None:
        year = se.group('year')
        month = se.group('month')
    else:
        raise ValueError('No date in file name')
    return year, month


def get_csv_file_from_dir(directory, relative=None):
    assert os.path.isdir(directory), 'Not a directory or not exist'
    directory = os.path.realpath(directory)
    files = glob.glob(os.path.join(directory, '*.csv'))
    if relative is not None:
        files = [os.path.relpath(f, relative) for f in files]
    return files


def filter_csv_file_by_time(files, **kwargs):
    if len(files) < 1:
        return
    try:
        year = kwargs.pop('year')
    except KeyError:
        year = None
    try:
        month = kwargs.pop('month')
    except KeyError:
        month = None

    if isinstance(month, int) and month < 10:
        month = '0' + str(month)

    if year is not None and month is not None:
        parser = re.compile(f'{year}-{month}\.csv')
    elif year is not None and month is None:
        parser = re.compile(f'{year}-\d\d\.csv')
    elif year is None and month is not None:
        parser = re.compile(f'\d\d\d\d-{month}\.csv')
    else:
        return
    results = []
    for ff in files:
        sr = parser.search(ff)
        if sr is not None:
            results.append(ff)
    return results


def read_parser_error(error):
    error = str(error)
    par = re.compile(r'Expected\s(?P<expect>\d+)\sfields\sin\sline\s(?P<line>\d+),\ssaw\s(?P<saw>\d+)')
    se = par.search(error)
    if se is None:
        return
    res = dict()
    res['expect'] = int(se.group('expect'))
    res['line'] = int(se.group('line'))
    res['saw'] = int(se.group('saw'))
    return res


def handle_parser_error(file: str, err_info: dict):
    file = open(file, 'r')
    lineno = read_parser_error(err_info).pop('line')
    try:
        line = next(islice(file, lineno-1, lineno))
    except StopIteration:
        raise TypeError(f'No line {lineno} exists in file')
    finally:
        file.close()
    return line


def set_destination(dest):
    global RAW_DIR
    if dest is None:
        return RAW_DIR
    assert os.path.isdir(dest)
    dest = os.path.realpath(dest)
    RAW_DIR = os.path.join(dest, 'data/raw')
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    return RAW_DIR


def get_download_path(source):
    html = requests.get(source).text
    soup = BeautifulSoup(html, 'lxml')
    csv_url = soup.find_all('key')

    data_parser = re.compile(r'trip\sdata/yellow_tripdata_\d\d\d\d-\d\d\.csv')
    zone_parser = re.compile(r'misc/taxi[+\s]_zone_lookup\.csv')
    csv_path = []
    zone_path = None
    for csv in csv_url:
        s = data_parser.search(csv.text)
        z = zone_parser.search(csv.text)
        if z is not None:
            zone_path = z.string.split(' ')
            zone_path = URL + '+'.join(zone_path)
        if s is not None:
            seg = s.string.split(' ')
            seg = '+'.join(seg)
            csv_path.append(URL + seg)
    return csv_path, zone_path


def download_file(url, target_dir):
    name = url.split('/')[-1]
    check_path = os.path.join(target_dir, name)
    if os.path.isfile(check_path) and os.path.getsize(check_path) > 0:
        print(name + '...Exists')
        return name
    print(name + '...Started')
    file = requests.get(url, stream=False)
    try:
        assert file.status_code == 200
    except AssertionError:
        with open(LOG, 'a') as f:
            f.write(name + f'...not downloaded: {file.status_code}-{file.reason}\n')
    else:
        try:
            with open(os.path.join(target_dir, name), 'wb') as f:
                f.write(file.content)
        except Exception as err:
            with open(LOG, 'a') as f:
                f.write(name + f'...{sys.exc_info()[0]}: {err}\n')
            print(name + f'...{sys.exc_info()[0]}: {err}')
        else:
            print(name + '...OK')
        return name


def download_file_parallel(num_core, destination=None):
    if destination is not None:
        set_destination(destination)
    csv_path, zone_path = get_download_path(URL)
    zone_name = download_file(zone_path, DATA_DIR)

    item = [(csv, RAW_DIR) for csv in csv_path]
    with mp.Pool(num_core) as pool:
        data_names = pool.starmap(download_file, item)

    return data_names, zone_name


if __name__ == '__main__':
    # rt = get_download_path(URL)
    # rt = filter_csv_file_by_time(rt[0], year=2010, month=3)
    # for ff in rt:
    #     download_file(ff, RAW_DIR)
    import pandas as pd
    f = open('data/raw/yellow_tripdata_2010-03.csv', 'r')
    try:
        pd.read_csv(f)
    except pd.errors.ParserError as err:
        f.close()
        f = open('data/raw/yellow_tripdata_2010-03.csv', 'r')
        print(err)
        dd = read_parser_error(err)
        l = handle_parser_error(f, dd)
        print(l)
        print(len(l.split(',')))
    f.close()
