from bs4 import BeautifulSoup
import requests
import re
import multiprocessing as mp
import time
import os

RAW_DIR = 'data/raw'
DATA_DIR = 'data/'
LOG_DIR = 'log/'
AAM_DIR = 'data/aam'
ATM_DIR = 'data/atm'

if not os.path.exists(RAW_DIR):
    os.makedirs(RAW_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists(AAM_DIR):
    os.makedirs(AAM_DIR)

if not os.path.exists(ATM_DIR):
    os.makedirs(ATM_DIR)

URL = "https://nyc-tlc.s3.amazonaws.com/"
LOG = os.path.join(LOG_DIR, f'download-{time.time()}.log')


def get_download_path(source):
    html = requests.get(source).text
    soup = BeautifulSoup(html, 'lxml')
    csv_url = soup.find_all('key')

    data_parser = re.compile(r'trip\sdata/yellow_tripdata_\d\d\d\d-\d\d\.csv')
    zone_parser = re.compile(r'misc/taxi[+\s]_zone_lookup.csv')
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


def download_file(url, target_dir, lock):
    name = url.split('/')[-1]
    file = requests.get(url, stream=False)
    lock.acquire()
    with open(LOG, 'a') as f:
        f.write(name + '...OK\n')
    print(name + '...OK\n')
    lock.release()

    try:
        assert file.status_code == 200
    except AssertionError:
        lock.acquire()
        with open(LOG, 'a') as f:
            f.write(name + f'...not downloaded: {file.status_code}-{file.reason}\n')
        lock.release()
    else:
        with open(os.path.join(target_dir, name), 'wb') as f:
            f.write(file.content)
        return name


def download_file_parallel(num_core):
    lk = mp.Lock()
    csv_path, zone_path = get_download_path(URL)

    zone_name = download_file(zone_path, DATA_DIR, lk)

    item = [(csv, RAW_DIR, lk) for csv in csv_path]
    with mp.Pool(num_core) as pool:
        data_names = pool.starmap(download_file, item)

    return data_names, zone_name


if __name__ == '__main__':
    print(get_download_path(URL)[1])
