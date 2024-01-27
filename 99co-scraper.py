import os
import time
import re
import cfscrape
import random
import requests
import argparse
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import date
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


def to_snake_case(input_string):
    # Replace spaces with underscores and convert to lowercase
    snake_case_string = input_string.replace(' ', '_').lower()
    return snake_case_string


class NinetyNineCoScraper:
    DISTRICTS = {
        "01": "Boat Quay / Raffles Place / Marina",
        "02": "Chinatown / Tanjong Pagar",
        "03": "Alexandra / Commonwealth",
        "04": "Harbourfront / Telok Blangah",
        "05": "Buona Vista / West Coast / Clementi",
        "06": "City Hall / Clarke Quay",
        "07": "Beach Road / Bugis / Rochor",
        "08": "Farrer Park / Serangoon Rd",
        "09": "Orchard / River Valley",
        "10": "Tanglin / Holland",
        "11": "Newton / Novena",
        "12": "Balestier / Toa Payoh",
        "13": "Macpherson / Potong Pasir",
        "14": "Eunos / Geylang / Paya Lebar",
        "15": "East Coast / Marine Parade",
        "16": "Bedok / Upper East Coast",
        "17": "Changi Airport / Changi Village",
        "18": "Pasir Ris / Tampines",
        "19": "Hougang / Punggol / Sengkang",
        "20": "Ang Mo Kio / Bishan / Thomson",
        "21": "Clementi Park / Upper Bukit Timah",
        "22": "Boon Lay / Jurong / Tuas",
        "23": "Bukit Batok / Bukit Panjang / Choa Chu Kang",
        "24": "Lim Chu Kang / Tengah",
        "25": "Admiralty / Woodlands",
        "26": "Mandai / Upper Thomson",
        "27": "Sembawang / Yishun",
        "28": "Seletar / Yio Chu Kang"
    }
    COLUMNS = ['property_name', 'listing_id', 'district', 'price', 'bedroom', 'bathroom', 'dimensions', 'address', 'price/sqft', 'floor_level',
               'furnishing', 'facing', 'built_year', 'tenure', 'property_type', 'url', 'facilities']

    def __init__(self,
                 header='https://www.99.co',
                 key='/singapore/rent',
                 query='?query_ids=dtdistrict{district}&query_type=district&rental_type=all',
                 ):
        self.header = header
        self.key = key
        self.query = query
        self.captcha_counter = 0
        self.rental_prices_file = f'./rental_prices/ninety_nine/{date.today()}.csv'
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
                print("=" * 50 + "\n")

                soup = BeautifulSoup(html_content, 'html.parser')

                if "captcha" in soup.text:
                    print('CAPTCHA -> Retrying ' +
                          '(' + str(trial) + '/20)...')
                    time.sleep(0.1)
                    self.captcha_counter += 1

                    # Wait 30s every 10 tries
                    if trial % 10 == 0:
                        print('Connection reset, retrying in 30 secs...')
                        time.sleep(30)
                    continue
                elif has_pages and not soup.select_one("ul.SearchPagination-links"):
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

    def pagination(self, soup):
        pagination = soup.select_one("ul.SearchPagination-links")
        try:
            # only 1 page
            if pagination.find_all("li",  class_="next disabled"):
                pages = int(pagination.find_all("a")[1].text)
            # grab the page number before the arrow
            else:
                pages = int(pagination.find_all("a")[-2].text)
        except AttributeError:
            if soup.find("h2", class_="name").text.split(' ')[2] == '0':
                print('No property found. Scraping stopped.')
                exit(0)
            exit(1)
        return pages

    def link_scraper(self, soup):
        links = []
        units = soup.find_all("div", class_="_2J3pS")
        for unit in units:
            prop = unit.find("a", itemprop='url')
            prop_name = prop['title'].strip()
            links.append((prop_name, prop["href"]))
        return links

    def get_prop_info(self, soup):
        output = {col_name: None for col_name in self.COLUMNS}

        try:
            output["price"] = soup.find(
                'div', id='price').find('p').text.strip()
        except Exception as err:
            print(f"Error scraping price: {err}")
            return {}

        try:
            beds_element = soup.find('img', {'alt': 'Beds'})
            beds = beds_element.find_next_sibling().text if beds_element else None

            baths_element = soup.find('img', {'alt': 'Bath'})
            baths = baths_element.find_next_sibling().text if baths_element else None

            floor_area_element = soup.find('img', {'alt': 'Floor Area'})
            floor_area = floor_area_element.find_next_sibling(
            ).text if floor_area_element else None

            output['bedroom'] = beds
            output['bathroom'] = baths
            output['dimensions'] = floor_area
        except Exception as err:
            print(f"Error scraping (bed,bath,sqft) info: {err}")

        try:
            address_span = soup.find('p', class_="dniCg _3j72o _2rhE-")
            address = address_span.text.strip().split('\n')[0]
            """ 
            e.g "· Executive Condo for Rent\nAdmiralty / Woodlands (D25)"
            -> Starts with "·" = no address
            -> remove everything after the first "·" until "Rent" and strip whitespace
            """
            if address.startswith('·'):
                raise Exception('Address not found')
            pattern = re.compile(r'\s·.*?Rent', re.DOTALL)
            address = re.sub(pattern, '', address)
            output['address'] = address.strip()
        except Exception as err:
            print(f"Error scraping address: {err}")

        try:
            # Extract all facilities
            facilities = soup.find_all('div', class_='_3atmT')
            res = []
            for facility in facilities:
                img_alt = facility.find('img')['alt']
                res.append(img_alt)

            output['facilities'] = res
        except Exception as err:
            print(f"Error scraping facilities: {err}")

        try:
            property_details_rows = soup.select(
                '#propertyDetails table._3NpKo tr._2dry3')

            """ 
            e.g
            Price/sqft: $7.5 psf
            Floor Level: High
            Furnishing: Fully
            Built year: 1976
            Tenure: 99-year leasehold
            Property type: Apartment Whole Unit
            """
            for row in property_details_rows:
                columns = row.find_all('td', class_='NomDX')
                values = row.find_all('td', class_='XCAFU')
                not_included = set(['Last updated'])

                for col, val in zip(columns, values):
                    label = col.get_text(strip=True)
                    if label in not_included:
                        continue

                    if label == 'Property type':
                        output[to_snake_case(
                            label)] = val.get_text()
                        continue

                    output[to_snake_case(
                        label)] = val.get_text(strip=True)
        except Exception as err:
            print(f"Error scraping property details: {err}")

        return output

    def scrape_rental_prices(self, district, debug):
        self.query = self.query.format(district=district)
        print(f"Scraping {self.DISTRICTS[district]}...")

        soup, pages = self.initial_fetch()
        # Scrape links from the first page for rental properties
        self.props += self.link_scraper(soup)
        print('\rPage 1/{} done.'.format(str(pages)))

        # Scrape subsequent pages
        for page in range(2, pages + 1):
            if debug:
                continue
            if self.captcha_counter == 19:
                with open('current_page.txt', 'w') as file:
                    file.write(str(page))
                break

            soup = self.fetch_html(self.header + self.key + '/?page_num=' +
                                   str(page) + self.query, True)
            if not soup:
                print(f'Error fetching page {page}, skipping...')
                continue
            self.props += self.link_scraper(soup)
            print('\rPage {}/{} done.'.format(str(page), str(pages)))

        # Scrape rental info for each property
        rental_infos = []
        print('\nA total of ' + str(min(100, len(self.props))) +
              ' properties will be scraped.\n')

        for i, prop in enumerate(self.props):
            if debug and i == 6:
                break
            # only scrape 100/district
            if i == 101:
                break
            print(f"Fetching {prop[0]}...")

            url = self.header + prop[1]
            prop_soup = self.fetch_html(url, False)
            rental_info = self.get_prop_info(prop_soup)
            if rental_info == {}:
                continue

            rental_info["property_name"] = prop[0]
            rental_info["district"] = self.DISTRICTS[district]
            rental_info["listing_id"] = url.split('-')[-1]
            rental_info["url"] = url
            rental_infos.append(rental_info)
            print(str(i + 1) + '/' + str(min(100, len(self.props))) + ' done!')

        df = pd.DataFrame(rental_infos)
        df = df[self.COLUMNS]
        print(df.head())
        self.output_to_csv(df)

        # reset for next run
        self.props = []
        self.query = '?query_ids=dtdistrict{district}&query_type=district&rental_type=all'

    def output_to_csv(self, df: pd.DataFrame) -> None:
        try:
            # Check if the CSV file exists
            file_exists = os.path.isfile(self.rental_prices_file)
            print(f"File exists: {file_exists}")
            # Open the CSV file in append mode if it exists, otherwise in write mode
            with open(self.rental_prices_file, 'a+' if file_exists else 'w', newline='') as file:
                # Write the header only if the file is newly created
                df.to_csv(file, index=False, header=not file_exists)

            if file_exists:
                print(f'Rental prices appended to {self.rental_prices_file}')
            else:
                print(f'Rental prices saved to {self.rental_prices_file}')
        except Exception as e:
            print(f'Error writing to CSV: {e}')

    def initial_fetch(self):
        soup = self.fetch_html(self.header + self.key + self.query, True)
        pages = min(10, self.pagination(soup))
        print(str(pages) + ' page will be scraped.\n')
        return soup, pages

    def print_title(self):
        print(
            '\n===================================================\n99co Rental Price Scraper v1.0\nAuthor: Rowen\n===================================================\n')
        time.sleep(2)
        print('Job initiated with query on rental properties in Singapore.')

    def run(self, debug):
        self.print_title()
        for district in self.DISTRICTS.keys():
            self.scrape_rental_prices(district, debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug mode")
    args = parser.parse_args()

    while True:
        try:
            start = time.time()
            ninetynine_co_scraper = NinetyNineCoScraper()
            ninetynine_co_scraper.run(debug=args.debug)
            print(f"Time taken: {time.time() - start} seconds")
            break
        except Exception:
            print('Error scraping, retrying...')
