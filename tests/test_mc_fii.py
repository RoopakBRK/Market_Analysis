import requests
from bs4 import BeautifulSoup
url = "https://www.moneycontrol.com/markets/fii-dii-data/"
headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text, 'html.parser')
div = soup.find('div', class_='fii_dii_summary')
if div:
    print(div.text[:500])
else:
    print("No summary div found. Let's look at all tables again.")
    print([table.get('class') for table in soup.find_all('table')])
