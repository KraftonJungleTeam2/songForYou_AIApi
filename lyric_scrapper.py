import requests
from bs4 import BeautifulSoup
import re

headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"}
url = "https://www.melon.com/search/song/index.htm"
song_url = "https://www.melon.com/song/detail.htm"

def get_lyric(title, artist):
    def str_norm(string: str):
        return ''.join(string.split()).lower()
    response = requests.get(url, {'q': ' '.join((title, artist))}, headers=headers)

    try:
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            
            song = soup.select_one('#cont_wrap .section_song #pageList table tbody tr .ellipsis a:last-child')
            artistid = soup.select_one('#cont_wrap .section_song #pageList table tbody tr #artistName a')
            
            songid = re.findall(r"'(.*?)'", song['href'].split(';')[0])[-1]

            title_melon = song['title']
            artist_melon = artistid.get_text()
        else:
            return None
    except: 
        return None

    if str_norm(title) != str_norm(title_melon):
        return None
    if str_norm(artist) != str_norm(artist_melon):
        return None
    
    response = requests.get(song_url, {'songId': songid}, headers=headers)

    try:
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            
            lyricid = soup.select_one("div.lyric")
            lyric = lyricid.get_text(separator="  ").strip()

            return lyric
        else:
            return None
    except:
        return None

if __name__ == "__main__":
    print(get_lyric("고백", "델리스파이스"))