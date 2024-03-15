import re
import time
import argparse
import logging
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from scraper import AbstractPropertyScraper

# suppress warnings
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


class PropnexScraper(AbstractPropertyScraper):
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

    def __init__(self,
                 header='https://www.propnex.com',
                 key='/rent',
                 query='?sortBy=newest&listingType=RENT&condoPropertyType=CONDO%2CAPT&district={district}',
                 ):
        super().__init__(header, key, query)
        self.platform_name = 'PropNex'
        self.pages_to_fetch = 3
        self.properties_per_page = 15
        self.pagination_element = 'div.listingPagination'
        self.property_card_listing_div_class = 'listing-box updated'
        self.rental_prices_dir = f'./rental_prices/propnex/'
        self.driver = self.init_driver()

    def init_driver(self):
        # Set up Chrome options for headless mode
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument("--log-level=3")

        # Start Chrome browser
        return webdriver.Chrome(options=chrome_options)

    def fetch_html(self, url, has_pages):
        try:
            for trial in range(1, 21):
                print('Loading ' + url)

                # Set up retry logic
                wait = WebDriverWait(self.driver, 10)

                # Make request
                self.driver.get(url)

                # Wait for page to load
                wait.until(EC.presence_of_element_located(
                    (By.TAG_NAME, 'body')
                ))

                # Get the page source
                html_content = self.driver.page_source

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

    def pagination(self, soup):
        pagination = soup.select_one(self.pagination_element)
        try:
            # only 1 page
            if len(pagination.find_all("a")) <= 1:
                pages = 1
            # grab the page number before the arrow
            else:
                pages = int(pagination.find_all("a")[-2].text)
        except Exception as err:
            print(pagination)
            raise Exception(f"Error scraping pagination: {err}")
        return pages

    def link_scraper(self, soup):
        links = []
        units = soup.find_all(
            "div", class_=self.property_card_listing_div_class)
        for unit in units:
            prop = unit.find("div", class_="listing-box-bottom").find("a")
            prop_name = prop.text.strip()
            links.append((prop_name, prop["href"]))
        return links

    def get_prop_info(self, soup):
        output = {col_name: None for col_name in self.COLUMNS}

        try:
            price_div = soup.find('p', class_="mt-2").text.strip().split(" (")
            output["price"] = price_div[0].strip()
            output["price/sqft"] = price_div[1].split(")")[0].strip()
        except Exception as err:
            print(f"Error scraping price: {err}")
            return {}

        try:
            bed_img = soup.find('img', src='/img/listing/ic_beds.png')

            beds = bed_img.parent.get_text(strip=True)
            output['bedroom'] = beds

        except Exception as err:
            print(f"Error scraping beds: {err}")

        try:
            bath_img = soup.find('img', src='/img/listing/ic_baths.png')

            baths = bath_img.parent.get_text(strip=True)
            output['bathroom'] = baths
        except Exception as err:
            print(f"Error scraping baths: {err}")

        try:
            property_boxes = soup.find_all('div', class_='property-list-box')
            not_included = set(['Property Group', 'Listing Type', 'District'])
            for box in property_boxes:
                labels = box.find_all('b')
                values = box.find_all('span')

                for col, val in zip(labels, values):
                    label = col.get_text(strip=True)
                    if label in not_included:
                        continue

                    if label == 'Floor Area':
                        output['dimensions'] = val.get_text(strip=True)
                        continue
                    elif label == 'Street Name':
                        output['address'] = val.get_text(strip=True)
                        continue
                    elif label == 'Floor':
                        output['floor_level'] = val.get_text(strip=True)
                        continue

                    output[PropnexScraper.to_snake_case(
                        label)] = val.get_text(strip=True)
        except Exception as err:
            print(f"Error scraping more details: {err}")

        try:
            location_map_box = soup.find('div', class_='location-map-box')
            iframe_src = location_map_box.find('iframe')['src']

            # Extract latitude and longitude using regular expression
            match = re.search(r'latLng:([0-9.-]+),([0-9.-]+)', iframe_src)

            if not match:
                print("Latitude and Longitude not found.")
                raise Exception("Latitude and Longitude not found.")

            output['latitude'] = match.group(1)
            output['longitude'] = match.group(2)
        except Exception as err:
            print(f"Error scraping latitude and longitude: {err}")

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

            soup = self.fetch_html(
                self.header + self.key + self.query + '&page_num=' + str(page), True)
            if not soup:
                print(f'Error fetching page {page}, skipping...')
                continue
            self.props += self.link_scraper(soup)
            print('\rPage {}/{} done.'.format(str(page), str(pages)))

        # Scrape rental info for each property
        rental_infos = []
        print('\nA total of ' + str(min(self.properties_per_page, len(self.props))) +
              ' properties will be scraped.\n')

        for i, prop in enumerate(self.props):
            if debug and i == 6:
                break
            # only scrape self.properties_per_page per district
            if i == self.properties_per_page + 1:
                break
            print(f"Fetching {prop[0]}...")

            url = self.header + prop[1]
            prop_soup = self.fetch_html(url, False)
            rental_info = self.get_prop_info(prop_soup)
            if rental_info == {}:
                continue

            rental_info["property_name"] = prop[0]
            rental_info["district"] = self.DISTRICTS[district]
            rental_info["listing_id"] = url.split('/')[-2]
            rental_info["url"] = url
            rental_infos.append(rental_info)
            print(str(i + 1) + '/' +
                  str(min(self.properties_per_page, len(self.props))) + ' done!')

        df = pd.DataFrame(rental_infos)
        if df.empty:
            print(
                f"No rental information found for {self.DISTRICTS[district]}")
            self.refresh_variables()
            return

        df = df[self.COLUMNS]
        print(df.head())
        self.output_to_csv(df)

        # reset for next run
        self.refresh_variables()

    def refresh_variables(self):
        self.props = []
        self.query = '?sortBy=newest&listingType=RENT&condoPropertyType=CONDO%2CAPT&district={district}'

    def __del__(self):
        self.driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug mode")
    args = parser.parse_args()

    while True:
        try:
            start = time.time()
            propnex_scraper = PropnexScraper()
            propnex_scraper.run(debug=args.debug)
            print(f"Time taken: {time.time() - start} seconds")
            break
        except Exception as err:
            print(
                f'Error scraping: {err.with_traceback(err.__traceback__)}\n\n, retrying...')
