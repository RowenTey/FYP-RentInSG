import argparse
import copy
import json
import logging
import re
import time
import traceback

import pandas as pd
from scraper import AbstractPropertyScraper


class PropertyGuruScraper(AbstractPropertyScraper):
    def __init__(
        self,
        header="https://www.propertyguru.com.sg",
        key="/property-for-rent",
        query="?market=residential&listing_type=rent&district_code[]=D01&search=true",
        use_proxies=False,
    ):
        super().__init__(header, key, query, use_proxies)
        self.pages_to_fetch = 15
        self.platform_name = "PropertyGuru"
        self.properties_per_page = 200
        self.pagination_element = "ul.pagination"
        self.rental_prices_dir = "./pkg/rental_prices/property_guru/"

        self.soup = None
        self.output = {}
        self.current_district = ""

    def pagination(self, soup):
        pagination = soup.select_one(self.pagination_element)
        try:
            # only 1 page
            if pagination.find_all("li", class_="pagination-next disabled"):
                pages = int(pagination.find_all("a")[1].text)
            # grab the page number before the arrow
            else:
                pages = int(pagination.find_all("a")[-2].text)
        except AttributeError:
            # TODO: check if this is the correct way to handle this
            if soup.find("h2", class_="name").text.split(" ")[2] == "0":
                logging.warning("No property found. Scraping stopped.")
            exit(1)
        return pages

    def link_scraper(self, soup):
        links = []
        units = soup.find_all(
            "div", class_="listing-card")
        logging.debug(f"Found {len(units)} units")
        for unit in units:
            prop = unit.find("a", itemprop="url")
            prop_name = prop["title"].strip()
            logging.debug(prop_name)
            links.append((prop_name, prop["href"]))
        return links

    def get_price(self) -> bool:
        try:
            price_h2 = self.soup.find("h2", class_="amount")
            price = price_h2.text.strip() if price_h2 else None

            if not price:
                raise Exception("Price not found")

            self.output["price"] = price
            return True
        except Exception as err:
            logging.warning(f"Error scraping price: {err}")
            logging.warning(self.html_content[0:100])
            return False

    def get_overview_items(self):
        try:
            bedrooms_element = self.soup.find(
                'i', class_='pgicon-bedroom').find_next_sibling('h4').text

            if bedrooms_element and bedrooms_element.lower() == "room":
                bedrooms = 1
            elif bedrooms_element:
                bedrooms = int(bedrooms_element.split(' ')[0])

            bathrooms = int(self.soup.find(
                'i', class_='pgicon-bathroom').find_next_sibling('h4').text.split(' ')[0])
            dimensions = int(self.soup.find('i', class_='pgicon-dimensions')
                             .find_next_sibling('h4').text.replace(',', '').split(' ')[0])

            self.output["bedroom"] = bedrooms
            self.output["bathroom"] = bathrooms
            self.output["dimensions"] = dimensions
        except Exception as err:
            logging.warning(f"Error scraping (bed,bath,sqft) info: {err}")

    def get_address(self):
        try:
            address = self.soup.find(
                'span', class_='full-address__address').text

            if f"({self.current_district})" in address:
                address = address.replace(
                    f"({self.current_district})", "").strip()

            if self.DISTRICTS[self.current_district] in address:
                address = address.replace(
                    self.DISTRICTS[self.current_district], "").strip()

            self.output["address"] = address
        except Exception as err:
            logging.warning(f"Error scraping address: {err}")

    def get_coordinates(self):
        try:
            # {"center": {"lat": 1.285299952, "lng": 103.8455397}}
            pattern = re.compile(
                r'"center":\s*{"lat":\s*([0-9.-]+),\s*"lng":\s*([0-9.-]+)}')
            match = pattern.search(self.html_content)

            if not match:
                raise Exception("Coordinates not found")

            json_string = "{" + match.group().replace("\\", "") + "}"
            json_data = json.loads(json_string)

            lat = json_data.get("center", {}).get("lat")
            lng = json_data.get("center", {}).get("lng")

            self.output["latitude"] = lat
            self.output["longitude"] = lng
        except Exception as err:
            logging.warning(f"Error scraping coordinates: {err}")

    def get_nearest_mrt(self):
        try:
            mrt_distance_element = self.soup.find(
                "span", class_="mrt-distance__text")
            mrt_distance = mrt_distance_element.text if mrt_distance_element else None
            if mrt_distance:
                mrt = mrt_distance.rsplit(" from ", 1)[-1]
                distance = mrt_distance \
                    .split("(", 1)[-1] \
                    .split(")", 1)[0] \
                    .replace(" m", "")
                distance = int(distance)

            self.output["nearest_mrt"] = mrt
            self.output["distance_to_nearest_mrt"] = distance
        except Exception as err:
            logging.warning(f"Error scraping nearest MRT: {err}")

    def get_facilities(self):
        try:
            facilities = self.soup.find_all(
                "div", class_="property-amenities__row-item")
            res = []
            for facility in facilities:
                facility_name = facility.text.strip()
                if not facility_name:
                    continue
                res.append(facility_name)

            if not res:
                raise Exception("Facilities not found")

            self.output["facilities"] = res
        except Exception as err:
            logging.warning(f"Error scraping facilities: {err}")

    def get_property_details(self):
        try:
            property_details_rows = self.soup.select("div.meta-table__item")

            label_map = {"psf": "price/sqft"}

            not_included = set(["Last updated"])
            for row in property_details_rows:
                label_div = row.find("div", class_="meta-table__item__label")
                value_div = row.find(
                    "div", class_="meta-table__item__value-text")

                if not label_div or not value_div:
                    continue

                label = label_div.get_text(strip=True)
                if label in not_included:
                    continue

                label_mapped = label_map.get(self.to_snake_case(
                    label.lower()), label.lower())

                self.output[label_mapped] = value_div.get_text(strip=True)
        except Exception as err:
            logging.warning(f"Error scraping property details: {err}")

    def get_prop_info(self):
        has_price = self.get_price()
        if not has_price:
            return {}

        self.get_overview_items()
        self.get_address()
        self.get_coordinates()
        self.get_nearest_mrt()
        self.get_facilities()
        self.get_property_details()

        logging.debug(self.output)
        return copy.deepcopy(self.output)

    def scrape_rental_prices(self, district, debug):
        self.query = self.query.format(district=district)
        self.current_district = district
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
                self.header + self.key + "/?page_num=" +
                str(page) + "&" + self.query[1:],
                True,
            )
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

        url = prop[1]
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
        rental_info["listing_id"] = url.split("-")[-1]
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
        self.query = "?market=residential&listing_type=rent&district_code[]={district}&search=true"
        self.props = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("-p", "--proxy", action="store_true",
                        help="Enable proxy mode")
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
        ninetynine_co_scraper = PropertyGuruScraper(use_proxies=args.proxy)
        ninetynine_co_scraper.run(debug=args.debug)
        logging.info(
            f"\nTime taken: {(time.time() - start) / 60 / 60 :.2f} hours")
    except Exception as err:
        traceback.print_exc()
        logging.warning(f"Error scraping - {err.__class__.__name__}: {err}\n")
