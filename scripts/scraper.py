import requests
import warnings
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# log = logging.getLogger("logger")
# log.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter("[INFO] - %(message)s")
# handler.setFormatter(formatter)
# log.addHandler(handler)

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
    return links

# print(scrape_site("https://www.mom.gov.sg/"))
