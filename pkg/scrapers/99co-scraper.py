import json
import time
import re
import argparse
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
        "28": "Seletar / Yio Chu Kang"
    }

    def __init__(self,
                 header='https://www.99.co',
                 key='/singapore/rent',
                 query='?query_ids=dtdistrict{district}&query_type=district&rental_type=all',
                 ):
        super().__init__(header, key, query)
        self.pages_to_fetch = 15
        self.platform_name = '99.co'
        self.properties_per_page = 200
        self.pagination_element = "ul.Pagination_SearchPagination_links__0JY7B"
        self.rental_prices_dir = f'./pkg/rental_prices/ninety_nine/'

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
            if soup.find("h2", class_="name").text.split(' ')[2] == '0':
                print('No property found. Scraping stopped.')
            exit(1)
        return pages

    def link_scraper(self, soup):
        links = []
        units = soup.find_all(
            "div", attrs={"data-cy": "listingCard"})
        for unit in units:
            prop = unit.find("a", itemprop='url')
            prop_name = prop['title'].strip()
            links.append((prop_name, prop["href"]))
        return links

    def get_prop_info(self, soup):
        output = {col_name: None for col_name in self.COLUMNS}

        try:
            price_div = soup.find(
                'div', id='price').find('p')
            price = price_div.text.strip() if price_div else None

            if not price:
                price_p = soup.find(
                    'p', class_='Heading_heading3__vJ885 Overview_text__TpBFy Overview_text__extra_bold__IdfcW Overview_text__lg__Hgcal Heading_baseColor__xWzRr')
                price = price_p.text.strip() if price_p else None

            if not price:
                raise Exception('Price not found')

            output['price'] = price
        except Exception as err:
            print(f"Error scraping price: {err}")
            print(self.html_content[0:100])
            return {}

        try:
            overview_items = soup.find_all(
                'div', class_='Overview_item__2NxRA')

            beds = baths = dimensions = None

            for item in overview_items:
                text = item.text
                if 'bed' in text.lower():
                    text = text.replace(' Beds', '').replace(
                        ' Bed', '').replace('(room)', '')
                    if '+' not in text:
                        beds = int(text.strip())
                        continue

                    num1, num2 = map(int, text.split('+'))
                    beds = num1 + num2
                elif 'bath' in text.lower():
                    baths = int(text.replace(
                        ' Baths', '').replace(' Bath', ''))
                elif 'sqft' in text.lower():
                    dimensions = int(text.replace(
                        ',', '').replace(' sqft', ''))

            output['bedroom'] = beds
            output['bathroom'] = baths
            output['dimensions'] = dimensions
        except Exception as err:
            print(f"Error scraping (bed,bath,sqft) info: {err}")

        try:
            address_element = soup.find(
                'span', class_='Overview_text__TpBFy Overview_text__underline__tINTE')
            address = address_element.text if address_element else None
            address_without_postcode = address.rsplit(
                ' ', 1)[0] if address else None
            output['address'] = address_without_postcode
        except Exception as err:
            print(f"Error scraping address: {err}")

        try:
            # example: {\"coordinates\":{\"lat\":1.2769696206188,\"lng\":103.8535109362695}
            pattern = re.compile(
                r'\\"coordinates\\":\{\\"lat\\":([0-9.-]+),\\"lng\\":([0-9.-]+)\}')

            match = None
            match = pattern.search(self.html_content)

            if not match:
                raise (Exception('Coordinates not found'))

            # Extract the matched JSON string
            json_string = '{' + match.group().replace('\\', '') + '}'

            # Load the JSON content
            json_data = json.loads(json_string)

            # Access lat and lng values
            lat = json_data.get('coordinates', {}).get('lat')
            lng = json_data.get('coordinates', {}).get('lng')

            output['latitude'] = lat
            output['longitude'] = lng
        except Exception as err:
            print(f"Error scraping coordinates: {err}")

        try:
            mrt_element = soup.find('a', class_='NearestMrt_link__mpgJ2')
            mrt = mrt_element.text if mrt_element else None
            if mrt:
                mrt = mrt.rsplit(' ', 1)[0]

            distance_element = soup.find_all(
                'span', class_='NearestMrt_text__13z7n')[-1]
            distance = distance_element.text if distance_element else None
            if distance:
                distance = distance.rsplit(' ', 1)[-1].replace('m', '')[1:-1]
                distance = int(distance)

            output['nearest_mrt'] = mrt
            output['distance_to_nearest_mrt'] = distance
        except Exception as err:
            print(f"Error scraping nearest MRT: {err}")
            print(mrt_element.text) if mrt_element else print(mrt_element)
            print(distance_element.text) if distance_element else print(
                distance_element)

        try:
            # Extract all facilities
            facilities = soup.find_all('div', class_='Amenities_grid__GMGLd')
            res = []
            for facility in facilities:
                img = facility.find('img')
                if not img:
                    continue
                res.append(img['alt'])

            if not res:
                raise Exception('Facilities not found')

            output['facilities'] = res
        except Exception as err:
            print(f"Error scraping facilities: {err}")

        try:
            property_details_rows = soup.select(
                'tr.KeyValueDescription_section__nPsI6'
            )

            """
            e.g
            Price/sqft: $7.5 psf
            Floor Level: High
            Furnishing: Fully
            Built year: 1976
            Tenure: 99-year leasehold
            Property type: Apartment Whole Unit
            """
            not_included = set(['Last updated'])
            for row in property_details_rows:
                columns = row.find_all(
                    'td', class_='KeyValueDescription_label__ZTXLo')
                values = row.find_all(
                    'td', class_='KeyValueDescription_text__wDVAb')

                for col, val in zip(columns, values):
                    label = col.get_text(strip=True)
                    if label in not_included:
                        continue

                    output[NinetyNineCoScraper.to_snake_case(
                        label)] = val.get_text(strip=True)

        except Exception as err:
            print(f"Error scraping property details: {err}")

        return output

    def scrape_rental_prices(self, district, debug):
        self.query = self.query.format(district=district)
        print(f"Scraping {self.DISTRICTS[district]}...")

        soup, pages = self.initial_fetch()
        if not soup:
            print(f'Error fetching initial page, skipping...')
            return
        # Scrape links from the first page for rental properties
        self.props += self.link_scraper(soup)
        print('\rPage 1/{} done.'.format(str(pages)))

        # Scrape subsequent pages
        for page in range(2, pages + 1):
            if debug:
                continue

            soup = self.fetch_html(self.header + self.key + '/?page_num=' +
                                   str(page) + '&' + self.query[1:], True)
            if not soup:
                print(f'Error fetching page {page}, skipping...')
                continue
            self.props += self.link_scraper(soup)
            print('\rPage {}/{} done.'.format(str(page), str(pages)))

        self.properties_per_page = self.properties_per_page if not debug else 1

        # Scrape rental info for each property
        rental_infos = []
        print('\nA total of ' + str(min(self.properties_per_page, len(self.props))) +
              ' properties will be scraped.\n')

        for i, prop in enumerate(self.props):
            # only scrape self.properties_per_page per district
            if i == self.properties_per_page:
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
            print(str(i + 1) + '/' +
                  str(min(self.properties_per_page, len(self.props))) + ' done!')

        df = pd.DataFrame(rental_infos)

        if df.empty:
            print(f"No properties found for {self.DISTRICTS[district]}")
            self.refresh_variables()
            return

        df = df[self.COLUMNS]
        print(df.head())
        self.output_to_csv(df)

        # reset for next run
        self.refresh_variables()

    def refresh_variables(self):
        self.props = []
        self.query = '?query_ids=dtdistrict{district}&query_type=district&rental_type=all'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug mode")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    try:
        start = time.time()
        ninetynine_co_scraper = NinetyNineCoScraper()
        ninetynine_co_scraper.run(debug=args.debug)
        print(f"\nTime taken: {time.time() - start} seconds")
    except Exception as err:
        print(
            f'Error scraping - {err.__class__.__name__}: {err}\nRetrying...')
