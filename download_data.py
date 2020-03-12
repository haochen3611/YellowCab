from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm

URL = "https://nyc-tlc.s3.amazonaws.com/"
html = requests.get(URL).text
soup = BeautifulSoup(html, 'lxml')
csv_url = soup.find_all('key')

parser = re.compile(r'trip\sdata/yellow_tripdata_\d\d\d\d-\d\d\.csv')
csv_path = []
for csv in csv_url:
    s = parser.search(csv.text)
    if s is not None:
        seg = s.string.split(' ')
        seg = '+'.join(seg)
        csv_path.append(seg)

print(csv_path)

ch_size = 256

for csv in csv_path:
    path = URL + csv
    file = requests.get(path, stream=True)
    with open('data/'+csv.split('/')[-1], 'wb') as f:
        itera = file.iter_content(chunk_size=ch_size)
        for ch in itera:
            f.write(ch)
    break
