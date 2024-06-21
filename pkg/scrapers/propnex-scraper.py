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


class PropnexScraper(AbstractPropertyScraper):
    def __init__(
        self,
        header="https://www.propnex.com",
        key="/rent",
        query="?sortBy=newest&listingType=RENT&district={district}",
    ):
        super().__init__(header, key, query)
        self.platform_name = "PropNex"
        self.pages_to_fetch = 15
        self.properties_per_page = 200
        self.pagination_element = "div.listingPagination"
        self.rental_prices_dir = f"./pkg/rental_prices/propnex/"

        self.soup = None
        self.output = {}

        self.driver = self.init_driver()

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
                logging.info("Loading " + url)

                # Make request
                time.sleep(random.randint(1, 5))
                wait = WebDriverWait(self.driver, 20)
                self.driver.get(url)

                # Wait for page to load
                wait.until(EC.presence_of_element_located(
                    (By.TAG_NAME, "body")))

                # Get the page source
                html_content = self.driver.page_source
                logging.info("=" * 75 + "\n")

                soup = BeautifulSoup(html_content, "html.parser")

                if "captcha" in soup.text:
                    self.handle_retry("CAPTCHA", trial)
                    continue

                if has_pages and not soup.select_one(self.pagination_element):
                    logging.info(self.html_content[:200])
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
            raise Exception(f"Error scraping pagination: {err}")
        return pages

    def link_scraper(self, soup):
        links = []
        units = soup.find_all("div", class_="listing-box updated")
        for unit in units:
            prop = unit.find("div", class_="listing-box-bottom").find("a")
            prop_name = prop.text.strip()
            links.append((prop_name, prop["href"]))
        return links

    def get_price(self,) -> bool:
        try:
            price_div = self.soup.find(
                "p", class_="mt-2").text.strip().split(" (")
            self.output["price"] = price_div[0].strip()
            self.output["price/sqft"] = price_div[1].split(")")[0].strip()
            return True
        except Exception as err:
            logging.info(f"Error scraping price: {err}")
            return False

    def get_overview_items(self):
        try:
            bed_img = self.soup.find("img", src="/img/listing/ic_beds.png")

            if bed_img is not None:
                beds = bed_img.parent.get_text(strip=True)
                self.output["bedroom"] = beds
            else:
                description = self.soup.find("div", class_="desc-box")
                self.output["bedroom"] = 1 if "studio" in \
                    description.text.lower() else None
        except Exception as err:
            logging.info(f"Error scraping beds: {err}")

        try:
            bath_img = self.soup.find("img", src="/img/listing/ic_baths.png")

            baths = bath_img.parent.get_text(strip=True)
            self.output["bathroom"] = baths
        except Exception as err:
            logging.info(f"Error scraping baths: {err}")

    def get_more_details(self):
        try:
            property_boxes = self.soup.find_all(
                "div", class_="property-list-box")
            not_included = set(["Property Group", "Listing Type", "District"])
            for box in property_boxes:
                labels = box.find_all("b")
                values = box.find_all("span")

                for col, val in zip(labels, values):
                    label = col.get_text(strip=True)
                    if label in not_included:
                        continue

                    if label == "Floor Area":
                        self.output["dimensions"] = val.get_text(strip=True)
                        continue
                    elif label == "Street Name":
                        self.output["address"] = val.get_text(strip=True)
                        continue
                    elif label == "Floor":
                        self.output["floor_level"] = val.get_text(strip=True)
                        continue

                    self.output[PropnexScraper.to_snake_case(
                        label)] = val.get_text(strip=True)
        except Exception as err:
            logging.info(f"Error scraping more details: {err}")

    def get_coordinates(self):
        try:
            location_map_box = self.soup.find("div", class_="location-map-box")
            iframe_src = location_map_box.find("iframe")["src"]

            # Extract latitude and longitude using regular expression
            match = re.search(r"latLng:([0-9.-]+),([0-9.-]+)", iframe_src)

            if not match:
                logging.info("Latitude and Longitude not found.")
                raise Exception("Latitude and Longitude not found.")

            self.output["latitude"] = match.group(1)
            self.output["longitude"] = match.group(2)
        except Exception as err:
            logging.info(f"Error scraping latitude and longitude: {err}")

    def get_nearest_mrt(self):
        try:
            mrt_element = self.soup.find("a", class_="NearestMrt_link__mpgJ2")
            mrt = mrt_element.text if mrt_element else None
            if mrt:
                mrt = mrt.rsplit(" ", 1)[0]

            distance_element = self.soup.find_all(
                "span", class_="NearestMrt_text__13z7n")[-1]
            distance = distance_element.text if distance_element else None
            if distance:
                distance = distance.rsplit(" ", 1)[-1].replace("m", "")[1:-1]
                distance = int(distance)

            self.output["nearest_mrt"] = mrt
            self.output["distance_to_nearest_mrt"] = distance
        except Exception as err:
            logging.warning(f"Error scraping nearest MRT: {err}")

    def get_facilities(self):
        try:
            # Find all 'li' elements with the class 'facilities-icons'
            facilities_section = self.soup.find(
                'section', class_='listing-fac-section')
            facilities_title_div = facilities_section.find(
                'div', class_='col-lg-3')
            facilities_ul_div = facilities_title_div.find_next_sibling('div')
            facilities_ul = facilities_ul_div.find('ul')

            facility_names = [
                facility.text.strip() for facility
                in facilities_ul.children if facility.text.strip()]

            logging.info(facility_names)
            self.output["facilities"] = facility_names
        except Exception as err:
            logging.warning(f"Error scraping facilities: {err}")

    def get_prop_info(self):
        has_price = self.get_price()
        if not has_price:
            return {}

        self.get_overview_items()
        self.get_more_details()
        self.get_coordinates()
        self.get_facilities()
        # self.get_nearest_mrt()

        logging.debug(self.output)
        return copy.deepcopy(self.output)

    def scrape_rental_prices(self, district, debug):
        self.query = self.query.format(district=district)
        logging.info(f"\n\nScraping {self.DISTRICTS[district]}...\n")

        soup, pages = self.initial_fetch()
        if not soup:
            logging.info(
                f"Error fetching initial page for {self.DISTRICTS[district]}, skipping...")
            return

        self.scrape_links(soup, pages, debug)
        self.scrape_properties(debug, district)

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

            soup = self.fetch_html(
                self.header + self.key + self.query + "&page_num=" + str(page), True)
            if not soup:
                logging.warning(f"Error fetching page {page}, skipping...")
                continue

            self.props += self.link_scraper(soup)
            logging.info(f"Page {page}/{pages} done.")

    def scrape_properties(self, debug, district):
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

            rental_info = self.scrape_property_info(prop, district)
            if rental_info:
                rental_infos.append(rental_info)

            logging.info(
                str(i + 1) + "/" + str(min(self.properties_per_page, len(self.props))) + " done!")

        self.create_dataframe(rental_infos, district)

    def scrape_property_info(self, prop, district):
        logging.info(f"Fetching {prop[0]}...")

        url = self.header + prop[1]
        prop_soup = self.fetch_html(url, False)
        if not prop_soup:
            logging.info(f"Error fetching {prop[0]}, skipping...")
            time.sleep(10)
            return

        self.soup = prop_soup
        self.output = {col_name: None for col_name in self.COLUMNS}

        rental_info = self.get_prop_info()
        if rental_info == {}:
            return

        rental_info["property_name"] = prop[0]
        rental_info["district"] = self.DISTRICTS[district]
        rental_info["listing_id"] = url.split("/")[-2]
        rental_info["url"] = url

        # reset for next property
        self.soup = None
        self.output.clear()

        return rental_info

    def create_dataframe(self, rental_infos, district):
        df = pd.DataFrame(rental_infos)

        if df.empty:
            logging.info(f"No properties found for {self.DISTRICTS[district]}")
            return

        df = df[self.COLUMNS]
        logging.info(f"\n{df.head()}")
        self.output_to_csv(df)

    def refresh_variables(self):
        self.props = []
        self.query = "?sortBy=newest&listingType=RENT&condoPropertyType=CONDO%2CAPT&district={district}"

    def __del__(self):
        self.driver.close()


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
        propnex_scraper = PropnexScraper()
        propnex_scraper.run(debug=args.debug)
        logging.info(f"Time taken: {time.time() - start} seconds")
    except Exception as err:
        logging.info(
            f"Error scraping: {err.with_traceback(err.__traceback__)}\n\n, retrying...")
