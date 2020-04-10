from urllib.request import urlopen
from bs4 import BeautifulSoup
import re


def get_current_value():
    """
    Parsing current index value of S&P500 from yahoo finance
    """
    url = "https://finance.yahoo.com/quote/%5EGSPC"
    html = urlopen(url)

    soup = BeautifulSoup(html, 'lxml')
    text = soup.get_text()

    m = re.search('\d,\d\d\d.\d\d', text)

    price_current_text = m[0].replace(',', '')
    return float(price_current_text)


if __name__ == '__main__':
    price_current = get_current_value()
    print(price_current)
