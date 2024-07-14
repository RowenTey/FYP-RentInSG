## Technical Documentation - NinetyNineCo Scraper

This document outlines the flow of the NinetyNineCo scraper script, focusing on its technical aspects.

### AbstractPropertyScraper Class

The script defines an abstract base class `AbstractPropertyScraper` that provides a foundation for scrapers targeting different property listing platforms. It specifies the following core functionalities:

- **`DISTRICTS` (dict):** A dictionary mapping district codes to their corresponding names.
- **`COLUMNS` (list):** A list containing column names for the scraped data (e.g., property name, price, address, etc.).
- **Initialization (`__init__` method):**

  - Takes `header` (str), `key` (str), and `query` (str) as input to define the base URL structure for scraping.
  - Initializes other scraper-specific attributes like `platform_name` (str), `properties_per_page` (int), `pages_to_fetch` (int), etc.

- **Abstract Methods:**
  - `fetch_html(url: str, has_pages: bool) -> BeautifulSoup` : Fetches the HTML content of a URL and handles potential retries in case of errors.
  - `pagination(soup: BeautifulSoup) -> int` : Extracts the total number of pages from the pagination element (requires implementation specific to the website structure).
  - `link_scraper(soup: BeautifulSoup) -> List[str]`: Scrapes the property links from the HTML content (requires implementation specific to the website structure).
  - `get_prop_info(soup: BeautifulSoup) -> Dict[str, Any]`: Extracts property information from an individual property page (requires implementation specific to the website structure).
  - `scrape_rental_prices(self, district: str, debug: bool) -> None`: Scrapes rental prices for a specific district (orchestrates the scraping process for listings within a district).

### NinetyNineCoScraper Class

This class inherits from `AbstractPropertyScraper` and implements the abstract methods specific to scraping rental listings from NinetyNineCo.

- **`DISTRICTS`:** Inherits the district dictionary from the base class.
- **Initialization (`__init__` method):** Inherits from the base class and sets the `platform_name` attribute to "NinetyNineCo".
- **`fetch_html` method:**
  - Uses `cloudscraper` for scraping with protections against anti-scraping measures.
  - Implements a retry mechanism with exponential backoff in case of fetching errors.
  - Checks for CAPTCHAs and includes logic for handling them (potentially requiring further development).
- **`pagination` method:** Needs to be implemented based on NinetyNineCo's pagination structure.
- **`link_scraper` method:** Needs to be implemented to extract property links from NinetyNineCo's listing pages.
- **`get_prop_info` method:** Needs to be implemented to extract property details from individual property pages on NinetyNineCo.
- **`scrape_rental_prices` method:**
  - Iterates through districts defined in `DISTRICTS`.
  - For each district:
    - Constructs the district-specific URL using `query` and `header` + `key`.
    - Calls `fetch_html` to get the initial HTML content.
    - Calls `pagination` to determine the number of pages to scrape (implementation required).
    - Iterates through listing pages (up to `pages_to_fetch` limit). For each listing page:
      - Calls `link_scraper` to extract property links.
      - (Optional) Limit the number of scraped properties per page using `properties_per_page`.
    - For each property link, potentially involves additional requests to scrape details using `get_prop_info` (implementation required).
  - Saves the scraped data in CSV format (implementation in `output_to_csv` method).

### Other methods

- `output_to_csv(df: pd.DataFrame) -> None`: Saves the scraped data in a CSV file named by the current date.
- `initial_fetch` method: Fetches the initial HTML content and determines the number of pages to scrape (calls `fetch_html` and `pagination`).
- `print_title` method: Prints the scraper title and introductory message.
- `check_for_failure` method: Sends notifications if scraping encounters excessive failures (requires notification library integration).
- `run(self, debug: bool) -> None` : The main entry point of the script.
  - Calls `print_title` to display scraper information.
  - Iterates through districts defined in `DISTRICTS`.
  - For each district, calls `scrape_rental_prices
