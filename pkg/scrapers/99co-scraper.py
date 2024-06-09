import argparse
import copy
import json
import logging
import re
import time
import traceback

import pandas as pd
from scraper import AbstractPropertyScraper


class NinetyNineCoScraper(AbstractPropertyScraper):
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
        "28": "Seletar / Yio Chu Kang",
    }

    def __init__(
        self,
        header="https://www.99.co",
        key="/singapore/rent",
        query="?query_ids=dtdistrict{district}&query_type=district&rental_type=all",
    ):
        super().__init__(header, key, query)
        self.pages_to_fetch = 15
        self.platform_name = "99.co"
        self.properties_per_page = 150
        self.pagination_element = "ul.Pagination_SearchPagination_links__0JY7B"
        self.rental_prices_dir = f"./pkg/rental_prices/ninety_nine/"

        self.soup = None
        self.output = {}

    def pagination(self, soup):
        pagination = soup.select_one(self.pagination_element)
        try:
            # only 1 page
            if pagination.find_all("li", class_="next disabled"):
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
        units = soup.find_all("div", attrs={"data-cy": "listingCard"})
        for unit in units:
            prop = unit.find("a", itemprop="url")
            prop_name = prop["title"].strip()
            links.append((prop_name, prop["href"]))
        return links

    def get_price(self) -> bool:
        try:
            price_div = self.soup.find("div", id="price").find("p")
            price = price_div.text.strip() if price_div else None

            if not price:
                price_p = self.soup.find(
                    "p",
                    class_="Heading_heading3__vJ885 Overview_text__TpBFy Overview_text__extra_bold__IdfcW Overview_text__lg__Hgcal Heading_baseColor__xWzRr",
                )
                price = price_p.text.strip() if price_p else None

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
            overview_items = self.soup.find_all(
                "div", class_="Overview_item__2NxRA"
            )

            beds = baths = dimensions = None

            for item in overview_items:
                text = item.text
                if "room" in text.lower() and "sqft" not in text.lower():
                    beds = 1
                elif "bed" in text.lower():
                    text = text.replace(" Beds", "").replace(" Bed", "")
                    if "+" not in text:
                        beds = int(text.strip())
                        continue

                    num1, num2 = map(int, text.split("+"))
                    beds = num1 + num2
                elif "bath" in text.lower():
                    baths = int(
                        text.replace(" Baths", "").replace(" Bath", "")
                    )
                elif "sqft" in text.lower():
                    dimensions = int(
                        text.replace(",", "")
                        .replace(" sqft", "")
                        .replace("(room)", "")
                    )

            self.output["bedroom"] = beds
            self.output["bathroom"] = baths
            self.output["dimensions"] = dimensions
        except Exception as err:
            logging.warning(f"Error scraping (bed,bath,sqft) info: {err}")

    def get_address(self):
        try:
            address_element = self.soup.find(
                "span",
                class_="Overview_text__TpBFy Overview_text__underline__tINTE",
            )
            address = address_element.text if address_element else None
            address_without_postcode = (
                address.rsplit(" ", 1)[0] if address else None
            )
            self.output["address"] = address_without_postcode
        except Exception as err:
            logging.warning(f"Error scraping address: {err}")

    def get_coordinates(self):
        try:
            pattern = re.compile(
                r'\\"coordinates\\":\{\\"lat\\":([0-9.-]+),\\"lng\\":([0-9.-]+)\}'
            )
            match = pattern.search(self.html_content)

            if not match:
                raise (Exception("Coordinates not found"))

            json_string = "{" + match.group().replace("\\", "") + "}"
            json_data = json.loads(json_string)

            lat = json_data.get("coordinates", {}).get("lat")
            lng = json_data.get("coordinates", {}).get("lng")

            self.output["latitude"] = lat
            self.output["longitude"] = lng
        except Exception as err:
            logging.warning(f"Error scraping coordinates: {err}")

    def get_nearest_mrt(self):
        try:
            mrt_element = self.soup.find("a", class_="NearestMrt_link__mpgJ2")
            mrt = mrt_element.text if mrt_element else None
            if mrt:
                mrt = mrt.rsplit(" ", 1)[0]

            distance_element = self.soup.find_all(
                "span", class_="NearestMrt_text__13z7n"
            )[-1]
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
            facilities = self.soup.find_all(
                "div", class_="Amenities_grid__GMGLd"
            )
            res = []
            for facility in facilities:
                img = facility.find("img")
                if not img:
                    continue
                res.append(img["alt"])

            if not res:
                raise Exception("Facilities not found")

            self.output["facilities"] = res
        except Exception as err:
            logging.warning(f"Error scraping facilities: {err}")

    def get_property_details(self):
        try:
            property_details_rows = self.soup.select(
                "tr.KeyValueDescription_section__nPsI6"
            )

            not_included = set(["Last updated"])
            for row in property_details_rows:
                columns = row.find_all(
                    "td", class_="KeyValueDescription_label__ZTXLo"
                )
                values = row.find_all(
                    "td", class_="KeyValueDescription_text__wDVAb"
                )

                for col, val in zip(columns, values):
                    label = col.get_text(strip=True)
                    if label in not_included:
                        continue

                    self.output[self.to_snake_case(label)] = val.get_text(
                        strip=True
                    )
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
        logging.info(f"\n\nScraping {self.DISTRICTS[district]}...\n")

        soup, pages = self.initial_fetch()
        if not soup:
            logging.info(
                f"Error fetching initial page for {self.DISTRICTS[district]}, skipping..."
            )
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
                self.header
                + self.key
                + "/?page_num="
                + str(page)
                + "&"
                + self.query[1:],
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
            "A total of "
            + str(min(self.properties_per_page, len(self.props)))
            + " properties will be scraped."
        )

        for i, prop in enumerate(self.props):
            # only scrape self.properties_per_page per district
            if i == self.properties_per_page:
                break

            rental_info = self.scrape_property_info(prop, district)
            if rental_info:
                rental_infos.append(rental_info)

            logging.info(
                str(i + 1)
                + "/"
                + str(min(self.properties_per_page, len(self.props)))
                + " done!"
            )

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
        self.query = "?query_ids=dtdistrict{district}&query_type=district&rental_type=all"
        self.props = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug mode"
    )
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
        ninetynine_co_scraper = NinetyNineCoScraper()
        ninetynine_co_scraper.run(debug=args.debug)
        logging.info(
            f"\nTime taken: {(time.time() - start) / 60 / 60 :.2f} hours"
        )
    except Exception as err:
        traceback.print_exc()
        logging.warning(f"Error scraping - {err.__class__.__name__}: {err}\n")
