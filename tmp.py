from selenium.webdriver.chrome.options import Options
from selenium import webdriver

url = 'https://github.com/'

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
options.add_argument("--remote-debugging-port=9222")

driver = webdriver.Chrome(options=options)

# Navigate to github.com
driver.get(url)

# Extract the top heading from github.com
text = driver.find_element_by_class_name('h000-mktg').text

print(text)
