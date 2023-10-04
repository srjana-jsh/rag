import requests
import pandas as pd
import numpy as np
import re
import time
import random
import warnings
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# Path: scraper.ipynb
def get_soup(url):
    """
    Returns a BeautifulSoup object from a given url.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html5lib")
    return soup

# Path: scraper.ipynb
def get_links(url):
    """
    Returns a list of links from a given url.
    """
    soup = get_soup(url)
    links = []
    for link in soup.find_all("a"):
        links.append(url + link.get("href")) if link.get("href") and link.get("href")[
            :4
        ] != "http" else links.append(link.get("href"))
    return links

# Path: scraper.ipynb
def get_links_from_list(url_list):
    """
    Returns a list of links from a given list of urls.
    """
    links = []
    for url in url_list:
        links.append(get_links(url))
    return links

# Path: scraper.ipynb
def scrape_site(url):
    links = get_links_from_list([url])[0]
    return links

# print(scrape_site("https://www.mom.gov.sg/"))
