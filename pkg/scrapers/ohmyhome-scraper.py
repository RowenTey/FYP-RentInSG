import argparse
import logging
import re
import random
import time
import copy

import pandas as pd
from bs4 import BeautifulSoup
from scraper import AbstractPropertyScraper
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# suppress warnings
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


class OhMyHomeScraper(AbstractPropertyScraper):
    def __init__(
        self,
        header="https://ohmyhome.com/",
        key="en-sg/listing-results/",
        query="?filterType=HOME_RENTAL_ALL",
    ):
        super().__init__(header, key, query)
        self.pages_to_fetch = 30
        self.platform_name = "Oh My Home"
        self.properties_per_page = 500
        self.pagination_element = "ul.MuiPagination-ul"
        self.rental_prices_dir = "./pkg/rental_prices/ohmyhome/"

        self.soup = None
        self.output = {}

        self.driver = self.init_driver()
        self.page_json = None

    def init_driver(self):
        # Set up Chrome options for headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # chrome_options.add_argument("window-size=1400,1500")

        # Start Chrome browser
        return webdriver.Chrome(options=chrome_options)

    def fetch_html(self, url, has_pages):
        try:
            for trial in range(1, 21):
                self.monitor_cpu()

                # Set up retry logic
                wait = WebDriverWait(self.driver, 10)
                logging.info("Loading " + url)
                time.sleep(random.randint(1, 5))
                wait = WebDriverWait(self.driver, 20)
                self.driver.get(url)

                # Wait for page to load
                wait.until(EC.presence_of_element_located(
                    (By.TAG_NAME, 'body')
                ))

                # Get the page source
                self.html_content = self.driver.page_source
                logging.info("=" * 75)

                soup = BeautifulSoup(self.html_content, "html.parser")

                if "captcha" in soup.text:
                    self.handle_retry("CAPTCHA", trial)
                    continue

                if has_pages and not soup.select_one(self.pagination_element):
                    self.handle_retry("No pages", trial)
                    continue

                if "No Results" in soup.text:
                    logging.info(f"Invalid URL: {url}, exiting...")
                    exit(1)

                return soup

            return None
        except Exception as err:
            logging.info(f"Error fetching HTML: {err}")
            return None

    def fetch_next_page(self, page_num):
        # find button with page_num in its text and click it then return the soup
        try:
            page_button = self.driver.find_element(
                By.XPATH, f"//button[text()='{page_num}']")
            page_button.click()

            for trial in range(1, 21):
                # Get the page source
                self.html_content = self.driver.page_source
                logging.info("=" * 75)

                soup = BeautifulSoup(self.html_content, "html.parser")

                if "captcha" in soup.text:
                    self.handle_retry("CAPTCHA", trial)
                    continue

                if not soup.select_one(self.pagination_element):
                    self.handle_retry("No pages", trial)
                    continue

                if "No Results" in soup.text:
                    logging.info(f"Invalid page: {page_num}, exiting...")
                    exit(1)

                return soup

            return None
        except Exception as err:
            logging.info(f"Error fetching next page: {err}")
            return None

    def pagination(self, soup):
        pagination = soup.select_one(self.pagination_element)
        try:
            buttons = pagination.find_all(
                "button", class_="MuiPaginationItem-root")
            logging.info(f"Found {buttons[-1].text} pages")
            pages = int(buttons[-1].text)
        except Exception as err:
            raise Exception(f"Error scraping pagination: {err}")
        return pages

    def link_scraper(self, soup):
        links = []
        units = soup.find("div", class_="css-tewpva")
        # iterate through its children
        for unit in units.children:
            if "href" not in unit.attrs:
                continue
            links.append(unit.attrs["href"])
        logging.info(f"Found {len(links)} links.")
        return links

    def get_price(self) -> bool:
        try:
            self.output["price"] = self.soup.find(
                "span", class_="MuiTypography-root MuiTypography-T5 css-mmxbx7").text.strip()
            logging.info(f"Scraped price: {self.output['price']}")
            return True
        except Exception as err:
            logging.info(f"Error scraping price: {err}")
            return False

    def get_prop_name(self):
        try:
            self.output["property_name"] = self.soup.find(
                "title").text.split("-")[0].strip()
            logging.info(
                f"Scraped property name: {self.output['property_name']}")
        except Exception as err:
            logging.info(f"Error scraping property name: {err}")

    def get_overview_items(self):
        try:
            bed_img = self.soup.find(
                "img", src="/assets/omh/listing/bedroom-logo.svg")
            beds = bed_img.find_parent(
                "span").next_sibling.get_text(strip=True)
            self.output["bedroom"] = beds
            logging.info(f"Scraped beds: {beds}")
        except Exception as err:
            logging.info(f"Error scraping beds: {err}")

        try:
            bath_img = self.soup.find(
                "img", src="/assets/omh/listing/toilet-logo.svg")
            baths = bath_img.find_parent(
                "span").next_sibling.get_text(strip=True)
            self.output["bathroom"] = baths
            logging.info(f"Scraped baths: {baths}")
        except Exception as err:
            logging.info(f"Error scraping baths: {err}")

        try:
            dimensions_img = self.soup.find(
                "img", src="/assets/omh/listing/size-logo.svg")
            dimensions = dimensions_img.find_parent(
                "span").next_sibling.get_text(strip=True)
            self.output["dimensions"] = dimensions
            logging.info(f"Scraped dimensions: {dimensions}")
        except Exception as err:
            logging.info(f"Error scraping dimensions: {err}")

    def get_more_details(self):
        try:
            property_boxes = self.soup.find_all(
                "div", class_="css-1mqofpb")

            not_included = set(
                ["Rental Type", "Lease Period", "Last Updated", "Floor Size", ])
            for box in property_boxes:
                label = box.find(
                    "span", class_="MuiTypography-root MuiTypography-Subtext2 css-8jgf7d")
                value = box.find(
                    "span", class_="MuiTypography-root MuiTypography-Subtext2 css-qs41mr")

                label = label.get_text(strip=True)
                val = value.get_text(strip=True)
                if label in not_included:
                    continue

                if label == "Property Type":
                    self.output["property_type"] = val
                elif label == "Tenure":
                    self.output["tenure"] = val
                elif label == "Project Name":
                    self.output["building_name"] = val
                elif label == "Completion":
                    self.output["built_year"] = val

                logging.info(f"Scraped {label}: {val}")
        except Exception as err:
            logging.info(f"Error scraping more details: {err}")

        try:
            floor_level_div = self.soup.find("div", class_="css-j7qwjs")
            self.output["floor_level"] = floor_level_div.get_text(
                strip=True).split(":")[-1]
            logging.info(f"Scraped floor level: {self.output['floor_level']}")
        except Exception as err:
            logging.info(f"Error scraping floor level: {err}")

        try:
            address_div = self.soup.find("div", class_="css-128dp9s")
            self.output["address"] = address_div.get_text(
                separator=" | ", strip=True)
            logging.info(f"Scraped address: {self.output['address']}")
        except Exception as err:
            logging.info(f"Error scraping address: {err}")

        try:
            description_div = self.soup.find(
                "div", class_="MuiBox-root css-hi4arb")
            description = description_div.get_text(strip=True)
            self.output["furnished"] = "Fully Furnished" \
                if "fully furnished" in description.lower() else "Unfurnished"
            logging.info(f"Scraped furnished: {self.output['furnished']}")
        except Exception as err:
            logging.info(f"Error scraping furnished: {err}")

    def get_coordinates(self):
        import json

        data = None
        try:
            next_data_script = self.soup.find("script", id="__NEXT_DATA__")
            json_data = next_data_script.get_text()

            data = json.loads(json_data)
            self.page_json = data

            self.output["latitude"] = data["props"]["pageProps"]["listing"]["address"]["latitude"]
            self.output["longitude"] = data["props"]["pageProps"]["listing"]["address"]["longitude"]

            logging.info(f"Scraped latitude: {self.output['latitude']}")
            logging.info(f"Scraped longitude: {self.output['longitude']}")
        except Exception as err:
            logging.info(f"Error scraping latitude and longitude: {err}")

        try:
            self.output["district"] = data["props"]["pageProps"]["listing"]["address"]["districtName"]

            logging.info(f"Scraped district: {self.output['district']}")
        except Exception as err:
            logging.info(f"Error scraping district: {err}")

    def get_facilities(self):
        try:
            if not self.page_json:
                raise Exception("No data found")

            self.output["facilities"] = self.page_json["props"]["pageProps"]["listing"]["facilities"]
            logging.info(f"Scraped facilities: {self.output['facilities']}")
        except Exception as err:
            logging.warning(f"Error scraping facilities: {err}")

    def get_prop_info(self):
        has_price = self.get_price()
        if not has_price:
            return {}

        self.get_coordinates()
        self.get_prop_name()
        self.get_overview_items()
        self.get_more_details()
        self.get_facilities()

        logging.debug(self.output)
        return copy.deepcopy(self.output)

    def scrape_rental_prices(self, debug):
        soup, pages = self.initial_fetch()
        if not soup:
            logging.info(
                f"Error fetching initial page, exiting...")
            return

        self.scrape_links(soup, pages, debug)
        self.scrape_properties(debug)

        # reset for next run
        self.refresh_variables()

    def scrape_links(self, soup, pages, debug):
        # Scrape links from the first page for rental properties
        self.props += self.link_scraper(soup)
        logging.info(f"Page 1/{pages} done.")

        # Scrape subsequent pages
        for page in range(2, pages + 1):
            # only scrape 1 page in debug mode
            if debug:
                continue

            soup = self.fetch_next_page(page)
            if not soup:
                logging.warning(f"Error fetching page {page}, skipping...")
                continue

            self.props += self.link_scraper(soup)
            logging.info(f"Page {page}/{pages} done.")

    def scrape_properties(self, debug):
        self.properties_per_page = self.properties_per_page if not debug else 1

        # Scrape rental info for each property
        rental_infos = []
        logging.info(
            "A total of " + str(min(self.properties_per_page,
                                len(self.props))) + " properties will be scraped."
        )

        for i, prop in enumerate(self.props):
            # only scrape self.properties_per_page per district
            if i == self.properties_per_page:
                break

            rental_info = self.scrape_property_info(prop)
            if rental_info:
                rental_infos.append(rental_info)

            logging.info(
                str(i + 1) + "/" + str(min(self.properties_per_page, len(self.props))) + " done!")

        self.create_dataframe(rental_infos)

    def scrape_property_info(self, prop):
        logging.info(f"Fetching property...")

        # remove leading '/' from url
        url = self.header + prop[1:]
        prop_soup = self.fetch_html(url, False)
        if not prop_soup:
            logging.info(f"Error fetching property, skipping...")
            time.sleep(10)
            return

        self.soup = prop_soup
        self.output = {col_name: None for col_name in self.COLUMNS}

        rental_info = self.get_prop_info()
        if rental_info == {}:
            return

        # rental_info["district"] = self.DISTRICTS[district]
        rental_info["listing_id"] = url.split("/")[-2]
        rental_info["url"] = url

        # reset for next property
        self.soup = None
        self.output.clear()
        self.page_json = None

        return rental_info

    def create_dataframe(self, rental_infos):
        df = pd.DataFrame(rental_infos)

        if df.empty:
            # logging.info(f"No properties found for {self.DISTRICTS[district]}")
            logging.info(f"No properties found.")
            return

        df = df[self.COLUMNS]
        logging.info(f"\n{df.head()}")
        self.output_to_csv(df)

    def refresh_variables(self):
        # self.query = "?sortBy=newest&listingType=RENT&condoPropertyType=CONDO%2CAPT&district={district}"
        self.props = []

    def run(self, debug):
        """
        Runs the scraper for all districts.

        Args:
            debug (bool): Indicates whether to print debug information.

        """
        self.print_title()
        self.scrape_rental_prices(debug)

        if not debug:
            self.check_for_failure()

    def __del__(self):
        self.driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug mode")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : %(filename)s-%(lineno)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        start = time.time()
        ohmyhome_scraper = OhMyHomeScraper()
        ohmyhome_scraper.run(debug=args.debug)
        logging.info(f"Time taken: {time.time() - start} seconds")
    except Exception as err:
        logging.info(
            f"Error scraping: {err.with_traceback(err.__traceback__)}\n\n, retrying...")
