from datetime import date
import os
import cfscrape
import random
import requests
import time
import pandas as pd
from abc import ABC, abstractmethod
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup


class AbstractPropertyScraper(ABC):
    DISTRICTS = {
        "D01": "Boat Quay / Raffles Place / Marina",
        "D02": "Chinatown / Tanjong Pagar",
        "D03": "Alexandra / Commonwealth",
        "D04": "Harbourfront / Telok Blangah",
        "D05": "Buona Vista / West Coast / Clementi",
        "D06": "City Hall / Clarke Quay",
        "D07": "Beach Road / Bugis / Rochor",
        "D08": "Farrer Park / Serangoon Rd",
        "D09": "Orchard / River Valley",
        "D10": "Tanglin / Holland",
        "D11": "Newton / Novena",
        "D12": "Balestier / Toa Payoh",
        "D13": "Macpherson / Potong Pasir",
        "D14": "Eunos / Geylang / Paya Lebar",
        "D15": "East Coast / Marine Parade",
        "D16": "Bedok / Upper East Coast",
        "D17": "Changi Airport / Changi Village",
        "D18": "Pasir Ris / Tampines",
        "D19": "Hougang / Punggol / Sengkang",
        "D20": "Ang Mo Kio / Bishan / Thomson",
        "D21": "Clementi Park / Upper Bukit Timah",
        "D22": "Boon Lay / Jurong / Tuas",
        "D23": "Bukit Batok / Bukit Panjang / Choa Chu Kang",
        "D24": "Lim Chu Kang / Tengah",
        "D25": "Admiralty / Woodlands",
        "D26": "Mandai / Upper Thomson",
        "D27": "Sembawang / Yishun",
        "D28": "Seletar / Yio Chu Kang"
    }
    COLUMNS = ['property_name', 'listing_id', 'district', 'price', 'bedroom', 'bathroom', 'dimensions', 'address', 'latitude', 'longitude', 'price/sqft', 'floor_level',
               'furnishing', 'facing', 'built_year', 'tenure', 'property_type', 'nearest_mrt', 'distance_to_nearest_mrt', 'url', 'facilities']

    def __init__(self, header, key, query):
        self.header = header
        self.key = key
        self.query = query
        self.platform_name = ''
        self.properties_per_page = 50  # default to 50 properties per page
        self.pages_to_fetch = 10  # default to 10 pages
        self.pagination_element = ''
        self.rental_price_dir = 'rental_prices/'
        self.props = []

    def fetch_html(self, url, has_pages):
        try:
            for trial in range(1, 21):
                print('Loading ' + url)
                ua = UserAgent()
                headers = {'User-Agent': ua.random}

                session = requests.Session()
                retry = Retry(connect=3, backoff_factor=0.5)
                adapter = HTTPAdapter(max_retries=retry)
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                session.headers.update(headers)
                scraper = cfscrape.create_scraper(sess=session)

                # print(scraper.get("http://httpbin.org/ip").json())
                # print(headers)

                time.sleep(random.randint(1, 3))
                html_content = scraper.get(url).text
                print("=" * 75 + "\n")

                soup = BeautifulSoup(html_content, 'html.parser')

                if "captcha" in soup.text:
                    print('CAPTCHA -> Retrying ' +
                          '(' + str(trial) + '/20)...')
                    time.sleep(0.1)

                    # Wait 30s every 10 tries
                    if trial % 10 == 0:
                        print('Connection reset, retrying in 30 secs...')
                        time.sleep(30)
                    continue
                elif has_pages and not soup.select_one(self.pagination_element):
                    print('No pages -> Retrying ' +
                          '(' + str(trial) + '/20)...')
                    time.sleep(0.1)

                    # Wait 30s every 10 tries
                    if trial % 10 == 0:
                        print('Connection reset, retrying in 30 secs...')
                        time.sleep(30)
                    continue
                elif "No Results" in soup.text:
                    print('Invalid URL: {url}, exiting...')
                    exit(1)

                return soup
            return None
        except Exception as err:
            print(f"Error fetching HTML: {err}")
            return None

    @abstractmethod
    def pagination(self, soup):
        pass

    @abstractmethod
    def link_scraper(self, soup):
        pass

    @abstractmethod
    def get_prop_info(self, soup):
        pass

    @abstractmethod
    def scrape_rental_prices(self, district, debug):
        pass

    def output_to_csv(self, df: pd.DataFrame) -> None:
        try:
            output_path = os.path.join(
                self.rental_prices_dir, f'{date.today()}.csv')

            # Check if the CSV file exists
            file_exists = os.path.isfile(output_path)

            # Open the CSV file in append mode if it exists, otherwise in write mode
            with open(output_path, 'a+' if file_exists else 'w', newline='') as file:
                # Write the header only if the file is newly created
                df.to_csv(file, index=False, header=not file_exists)

            if file_exists:
                print(f'Rental prices appended to {output_path}')
            else:
                print(f'Rental prices saved to {output_path}')
        except Exception as e:
            print(f'Error writing to CSV: {e}')

    def initial_fetch(self):
        soup = self.fetch_html(self.header + self.key + self.query, True)
        pages = min(self.pages_to_fetch, self.pagination(soup))
        print(str(pages) + ' page will be scraped.\n')
        return soup, pages

    def print_title(self):
        print(
            f'\n===================================================\n{self.platform_name} Rental Price Scraper v1.0\nAuthor: Rowen\n===================================================\n')
        time.sleep(2)
        print('Job initiated with query on rental properties in Singapore.')

    def run(self, debug):
        self.print_title()
        for district in self.DISTRICTS.keys():
            self.scrape_rental_prices(district, debug)

    @staticmethod
    def to_snake_case(input_string):
        # Replace spaces with underscores and convert to lowercase
        snake_case_string = input_string.replace(' ', '_').lower()
        return snake_case_string
