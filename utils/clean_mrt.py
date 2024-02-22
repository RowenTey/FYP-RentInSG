import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_mrt_data():
    resp = requests.get("https://mrtmapsingapore.com/mrt-stations-singapore/")
    soup = BeautifulSoup(resp.content, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')
    headers = [th.text.strip() for th in rows[0].find_all('th')]

    data = []
    for row in rows[1:]:
        cols = row.find_all(['td', 'th'])
        row_data = [col.text.strip() for col in cols]
        data.append(row_data)

    df = pd.DataFrame(data, columns=headers)
    return df


scrape_mrt_data()
