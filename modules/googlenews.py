from bs4 import BeautifulSoup
import requests

class GoogleNews:

    def __init__(self) -> None:
        self.BASE_URL = 'https://news.google.com/rss'

    def search(self, query: str, when = None):
        
        if when:
            query += ' when:{}'.format(when)
        
        url = '{}/search?q={}'.format(self.BASE_URL, query)

        content =  requests.get(url).content
        soup = BeautifulSoup(content)

        data = {'entries': []}

        for item in soup.find_all('item'):

            title = item.title.text
            link = item.link.next_sibling
            pubdate = item.pubdate.text

            row = {'title': title, 'link': link, 'pubdate': pubdate}

            data['entries'].append(row)

        return data
        