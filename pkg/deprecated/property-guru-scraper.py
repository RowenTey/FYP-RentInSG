import os
import re
import time
import cfscrape
import random
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import date
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


class PropertyGuruScraper:
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
        'property_name',
        'listing_id',
        'district',
        'price',
        'bedroom',
        'bathroom',
        'dimensions',
        'price/sqft',
        'tenure',
        'furnished',
        'address',
        'property_type',
        'floor_level',
        'url',
        'unit_features',
        'facilities',
    ]

    def __init__(self,
                 header='https://www.propertyguru.com.sg',
                 key='/property-for-rent',
                 query='?market=residential&property_type=all&listing_type=rent&district_code[]={district}&search=true'
                 ):
        self.header = header
        self.key = key
        self.query = query
        self.captcha_counter = 0
        self.rental_prices_file = f'./rental_prices/property_guru/{date.today()}.csv'
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
                print("=" * 50)

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
                elif has_pages and not soup.select_one("ul.pagination"):
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
        pagination = soup.select_one("ul.pagination")
        try:
            # only 1 page
            if pagination.find_all("li", class_="pagination-next disabled"):
                pages = int(pagination.find_all("a")[0]['data-page'])
            # grab the page number before the arrow
            else:
                pages = int(pagination.find_all("a")[-2]['data-page'])
        except AttributeError:
            if soup.find("h1", class_="title search-title").text.split(' ')[2] == '0':
                print('No property found. Scraping stopped.')
                exit(0)
            else:
                exit(1)
        return pages

    def link_scraper(self, soup):
        links = []
        units = soup.find_all("div", class_="listing-card")
        for unit in units:
            prop = unit.find("a", class_="nav-link")
            prop_name = prop['title'].split("-")[1].strip()
            links.append((prop_name, prop["href"]))
        return links

    def get_prop_info(self, soup, district):
        output = {col_name: None for col_name in self.COLUMNS}

        try:
            output["price"] = soup.find(
                "h2", class_="amount").text.split('/')[0].strip()
        except Exception as err:
            print(f"Error scraping price: {err}")
            return {}

        try:
            amenities_section = soup.find('div', class_='amenities')
            if amenities_section:
                for amenity_div in amenities_section.find_all('div', class_='amenity'):
                    icon_class = amenity_div.find('i')['class'][-1]
                    info_name = icon_class.split('-')[1]
                    text = amenity_div.find('h4', class_='amenity__text').text
                    output[info_name] = text
        except Exception as err:
            print(f"Error scraping (bed,bath,sqft) info: {err}")

        try:
            address_div = soup.find('div', class_='full-address')
            address = address_div.find(
                'span', class_='full-address__address').text.strip()
            tail = f"{self.DISTRICTS[district]} ({district})"
            if address.endswith(tail):
                address = address[:-len(tail)].strip()
            output["address"] = address
        except Exception as err:
            print(f"Error scraping address: {err}")

        try:
            unit_features = soup.find(
                'div', {'id': 'react-aria-:R2db0ud6:-tabpane-Unit Features'})
            facilities = soup.find(
                'div', {'id': 'react-aria-:R2db0ud6:-tabpane-Facilities'})

            unit_features_list = [feature.text.strip() for feature in unit_features.find_all(
                'div', class_='property-amenities__row-item')]
            facilities_list = [facility.text.strip() for facility in facilities.find_all(
                'div', class_='property-amenities__row-item')]

            output["unit_features"] = unit_features_list
            output["facilities"] = facilities_list
        except Exception as err:
            print(f"Error scraping unit features and facilities: {err}")

        try:
            more_details_section = soup.find(
                'h2', class_='meta-table__title', string='More details')
            if more_details_section:
                table = more_details_section.find_next('table')
                rows = table.find_all('tr', class_='row')

                for row in rows:
                    label = row.find('div', class_='meta-table__item__label')
                    value = row.find(
                        'div', class_='meta-table__item__value-text')

                    label_text = label.text.strip()
                    value_text = value.text.strip()

                    if label_text == 'Property Type':
                        output["property_type"] = value_text.split(' ')[0]
                    elif label_text == 'Floor Level':
                        output["floor_level"] = value_text
                    elif label_text == 'PSF':
                        output["price/sqft"] = value_text
                    elif label_text == 'Tenure':
                        output["tenure"] = value_text
                    elif label_text == 'Listing ID':
                        output["listing_id"] = value_text
        except Exception as err:
            print(f"Error scraping more details: {err}")

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
        print('\nA total of ' + str(min(40, len(self.props))) +
              ' properties will be scraped.\n')

        for i, prop in enumerate(self.props):
            if debug and i == 6:
                break
            # only scrape 40/district
            if i == 41:
                break
            print(f"Fetching {prop[0]}...")

            url = prop[1]
            prop_soup = self.fetch_html(url, False)
            rental_info = self.get_prop_info(prop_soup, district)
            if rental_info == {}:
                continue

            rental_info["property_name"] = prop[0]
            rental_info["district"] = self.DISTRICTS[district]
            rental_info["url"] = url
            rental_info["listing_id"] = url.split(
                '-')[-1] if not rental_info["listing_id"] else rental_info["listing_id"]
            rental_infos.append(rental_info)
            print(str(i + 1) + '/' + str(min(40, len(self.props))) + ' done!')

        df = pd.DataFrame(rental_infos)
        df = df[self.COLUMNS]
        print(df.head())
        self.output_to_csv(df)

        # reset for next run
        self.props = []
        self.query = '?market=residential&property_type=all&listing_type=rent&district_code[]={district}&search=true'

    def output_to_csv(self, df: pd.DataFrame) -> None:
        try:
            # Check if the CSV file exists
            file_exists = os.path.isfile(self.rental_prices_file)
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
            '\n===================================================\nPropertyGuru Rental Price Scraper v1.0\nAuthor: Rowen\n===================================================\n')
        time.sleep(2)
        print('Job initiated with query on rental properties in Singapore.')

    def run(self):
        self.print_title()
        for district in self.DISTRICTS.keys():
            self.scrape_rental_prices(district, False)


if __name__ == "__main__":
    start = time.time()
    property_guru_scraper = PropertyGuruScraper()
    property_guru_scraper.run()
    print(f"Time taken: {time.time() - start} seconds")
