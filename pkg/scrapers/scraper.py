import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

import random
import time
import tls_client
import pandas as pd
from datetime import date
from abc import ABC, abstractmethod
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from utils.notify import send_message
from dotenv import load_dotenv
load_dotenv()


class AbstractPropertyScraper(ABC):
    """
    Abstract base class for property scrapers.

    Attributes:
        DISTRICTS (dict): A dictionary mapping district codes to district names.
        COLUMNS (list): A list of column names for the scraped data.
        header (str): The header of the URL to scrape.
        key (str): The key of the URL to scrape.
        query (str): The query of the URL to scrape.
        platform_name (str): The name of the platform being scraped.
        properties_per_page (int): The number of properties to scrape per page.
        pages_to_fetch (int): The number of pages to fetch.
        pagination_element (str): The CSS selector for the pagination element.
        rental_price_dir (str): The directory to save the rental prices.
        props (list): A list to store the scraped properties.

    Methods:
        fetch_html(url: str, has_pages: bool) -> BeautifulSoup: Fetches the HTML content of a URL.
        pagination(soup: BeautifulSoup) -> int: Extracts the number of pages from the pagination element.
        link_scraper(soup: BeautifulSoup) -> List[str]: Scrapes the property links from the HTML content.
        get_prop_info(soup: BeautifulSoup) -> Dict[str, Any]: Extracts the property information from the HTML content.
        scrape_rental_prices(district: str, debug: bool) -> None: Scrapes rental prices for a specific district.
        output_to_csv(df: pd.DataFrame) -> None: Outputs the scraped data to a CSV file.
        initial_fetch() -> Tuple[BeautifulSoup, int]: Fetches the initial HTML content and determines the number of pages to scrape.
        print_title() -> None: Prints the title of the scraper.
        run(debug: bool) -> None: Runs the scraper for all districts.
        to_snake_case(input_string: str) -> str: Converts a string to snake case.

    """

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
    COLUMNS = [
        'property_name', 'listing_id', 'district', 'price', 'bedroom', 'bathroom', 'dimensions', 'address', 'latitude', 'longitude', 'price/sqft', 'floor_level',
        'furnishing', 'facing', 'built_year', 'tenure', 'property_type', 'nearest_mrt', 'distance_to_nearest_mrt', 'url', 'facilities'
    ]

    def __init__(self, header: str, key: str, query: str):
        """
        Initializes an AbstractPropertyScraper object.

        Args:
            header (str): The header of the URL to scrape.
            key (str): The key of the URL to scrape.
            query (str): The query of the URL to scrape.

        """
        self.header = header
        self.key = key
        self.query = query
        self.html_content = ''
        self.platform_name = ''
        self.properties_per_page = 50  # default to 50 properties per page
        self.pages_to_fetch = 10  # default to 10 pages
        self.pagination_element = ''
        self.property_card_listing_div_class = ''
        self.rental_price_dir = 'rental_prices/'
        self.failure_counter = 0
        self.props = []

    def fetch_html(self, url: str, has_pages: bool) -> BeautifulSoup:
        """
        Fetches the HTML content of a URL.

        Args:
            url (str): The URL to fetch.
            has_pages (bool): Indicates whether the URL has multiple pages.

        Returns:
            BeautifulSoup: The parsed HTML content.

        """
        try:
            for trial in range(1, 21):
                print('Loading ' + url)
                ua = UserAgent()
                headers = {'User-Agent': ua.random}

                # print(scraper.get("http://httpbin.org/ip").json())
                # print(headers)

                session = tls_client.Session(
                    client_identifier="chrome112",
                    random_tls_extension_order=True
                )

                time.sleep(random.randint(1, 3))
                # self.html_content = scraper.get(url).text
                self.html_content = session.get(url, headers=headers).text
                print("=" * 75 + "\n")

                soup = BeautifulSoup(self.html_content, 'html.parser')

                if "captcha" in soup.text:
                    print('CAPTCHA -> Retrying ' +
                          '(' + str(trial) + '/20)...')
                    time.sleep(0.1)
                    self.failure_counter += 1

                    # Wait 30s every 10 tries
                    if trial % 10 == 0:
                        print('Connection reset, retrying in 30 secs...')
                        time.sleep(30)
                    continue
                elif has_pages and not soup.select_one(self.pagination_element):
                    print('No pages -> Retrying ' +
                          '(' + str(trial) + '/20)...')
                    time.sleep(0.1)
                    self.failure_counter += 1

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
        """
        Extracts the number of pages from the pagination element.

        Args:
            soup (BeautifulSoup): The parsed HTML content.

        Returns:
            int: The number of pages.

        """
        pass

    @abstractmethod
    def link_scraper(self, soup):
        """
        Scrapes the property links from the HTML content.

        Args:
            soup (BeautifulSoup): The parsed HTML content.

        Returns:
            List[str]: A list of property links.

        """
        pass

    @abstractmethod
    def get_prop_info(self, soup):
        """
        Extracts the property information from the HTML content.

        Args:
            soup (BeautifulSoup): The parsed HTML content.

        Returns:
            Dict[str, Any]: A dictionary containing the property information.

        """
        pass

    @abstractmethod
    def scrape_rental_prices(self, district, debug):
        """
        Scrapes rental prices for a specific district.

        Args:
            district (str): The district code.
            debug (bool): Indicates whether to print debug information.

        """
        pass

    def output_to_csv(self, df: pd.DataFrame) -> None:
        """
        Outputs the scraped data to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame containing the scraped data.

        """
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

    def initial_fetch(self) -> tuple[BeautifulSoup, int]:
        """
        Fetches the initial HTML content and determines the number of pages to scrape.

        Returns:
            Tuple[BeautifulSoup, int]: The parsed HTML content and the number of pages.

        """
        soup = self.fetch_html(self.header + self.key + self.query, True)
        if not soup:
            print(f'Error fetching initial page, exiting...')
            return None, None
        pages = min(self.pages_to_fetch, self.pagination(soup))
        print(str(pages) + ' pages will be scraped.\n')
        return soup, pages

    def print_title(self):
        """
        Prints the title of the scraper.

        """
        print(
            f'\n===================================================\n{self.platform_name} Rental Price Scraper v1.0\nAuthor: Rowen\n===================================================\n')
        print('Job initiated with query on rental properties in Singapore.')

    def check_for_failure(self):
        if self.failure_counter >= 100:
            send_message(
                f"{self.platform_name} Scraper", "Exceeded 100 failures, please check!")

        csv_filepath = os.path.join(
            self.rental_prices_dir, f'{date.today()}.csv')
        try:
            df = pd.read_csv(csv_filepath)
        except Exception as e:
            send_message(
                f"{self.platform_name} Scraper", f"Error reading {csv_filepath} - No data for {date.today()}!")
            return

        if df.empty:
            send_message(
                f"{self.platform_name} Scraper", f"No data for {date.today()}!")
            return

        for column in df.columns:
            null_percentage = df[column].isnull().sum() / len(df) * 100
            if null_percentage > 50:
                send_message(f"{self.platform_name} Scraper",
                             f"Null values in column {column} exceed 50%!")

    def run(self, debug):
        """
        Runs the scraper for all districts.

        Args:
            debug (bool): Indicates whether to print debug information.

        """
        self.print_title()
        for district in self.DISTRICTS.keys():
            self.scrape_rental_prices(district, debug)

        self.check_for_failure()

    @staticmethod
    def to_snake_case(input_string: str) -> str:
        """
        Converts a string to snake case.

        Args:
            input_string (str): The input string.

        Returns:
            str: The string converted to snake case.

        """
        return input_string.replace(' ', '_').lower()
