import requests
import warnings
import os
import logging
from datetime import datetime
from bs4 import BeautifulSoup

#Logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(
    f"logs/{datetime.now().strftime('web_scraping_%H_%M_%d_%m_%Y.log')}",
    mode="a", 
    encoding="utf-8"        
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[INFO] - %(asctime)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Path: scraper.ipynb
def get_soup(url, header_template):
    """
    Returns a BeautifulSoup object from a given url.
    """
    response = requests.get(url, headers=header_template)
    soup = BeautifulSoup(response.text, "html5lib")
    return soup

# Path: scraper.ipynb
def get_links(url, header_template):
    """
    Returns a list of links from a given url.
    """
    soup = get_soup(url, header_template)
    links = []
    for link in soup.find_all("a"):
        links.append(url + link.get("href")) if link.get("href") and link.get("href")[
            :4
        ] != "http" else links.append(link.get("href"))
    return links

# Path: scraper.ipynb
def get_links_from_list(url_list, header_template):
    """
    Returns a list of links from a given list of urls.
    """
    links = []
    for url in url_list:
        links.extend(get_links(url, header_template))
    return links

# Path: scraper.ipynb
def scrape_site(url, header_template):
    links = list(set(get_links_from_list([url], header_template)))
    logger.info(f'List of links {links}')
    return links

# print(scrape_site("https://www.mom.gov.sg/"))
