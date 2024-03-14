import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from coordinates import fetch_coordinates
import re


def fetch_mall_info():
    url = "https://en.wikipedia.org/wiki/List_of_shopping_malls_in_Singapore"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    divs = soup.find_all('div', {'class': 'div-col'})

    li_elements = []
    for div in divs:
        li_elements.extend(div.find_all('li'))
    malls = [re.sub(r'\[\d+\]', '', li.text) for li in li_elements]
    df = pd.DataFrame(malls, columns=['name'])

    with ThreadPoolExecutor() as executor:
        future_to_coords = {executor.submit(
            fetch_coordinates, name): name for name in df['name']}
        for future in as_completed(future_to_coords):
            result = future.result()
            if result is None:
                continue

            school_name, coords = result
            df.loc[df['name'] ==
                   school_name, 'latitude'] = coords[0]
            df.loc[df['name'] ==
                   school_name, 'longitude'] = coords[1]

    return df[['name', 'latitude', 'longitude']]


if __name__ == '__main__':
    df = fetch_mall_info()
    df.drop(df[df['latitude'].isna()].index, inplace=True)
    df["mall_id"] = range(1, len(df) + 1)
    df = df[["mall_id", "name", "longitude", "latitude"]]
    print(df)

    from motherduckdb_connector import connect_to_motherduckdb
    db = connect_to_motherduckdb()
    db.insert_df("mall_info", df)
    db.close()
