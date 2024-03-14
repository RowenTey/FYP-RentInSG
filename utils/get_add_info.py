import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from coordinates import fetch_coordinates


def fetch_primary_school_info():
    # Fetch primary school info from data.gov.sg
    url = "https://en.wikipedia.org/wiki/List_of_primary_schools_in_Singapore"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    df = pd.read_html(str(table))[0]

    with ThreadPoolExecutor() as executor:
        future_to_coords = {executor.submit(
            fetch_coordinates, name): name for name in df['Name']}
        for future in as_completed(future_to_coords):
            result = future.result()
            if result is None:
                continue

            school_name, coords = result
            df.loc[df['Name'] ==
                   school_name, 'latitude'] = coords[0]
            df.loc[df['Name'] ==
                   school_name, 'longitude'] = coords[1]

    df.rename(columns={'Name': 'name', 'Area[3]': 'area'}, inplace=True)
    return df[['name', 'area', 'latitude', 'longitude']]


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
    df = fetch_primary_school_info()
    df.drop(df[df['latitude'].isna()].index, inplace=True)
    df["school_id"] = range(1, len(df) + 1)
    df = df[["school_id", "name", "area", "longitude", "latitude"]]
    print(df)

    from motherduckdb_connector import connect_to_motherduckdb
    db = connect_to_motherduckdb()
    db.insert_df("primary_school_info", df)
    db.close()

    # Fetch mall
    # df = fetch_mall_info()
    # df.drop(df[df['latitude'].isna()].index, inplace=True)
    # df["mall_id"] = range(1, len(df) + 1)
    # df = df[["mall_id", "name", "longitude", "latitude"]]
    # print(df)

    # from motherduckdb_connector import connect_to_motherduckdb
    # db = connect_to_motherduckdb()
    # db.insert_df("mall_info", df)
    # db.close()
