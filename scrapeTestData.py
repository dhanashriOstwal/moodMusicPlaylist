# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:33:03 2016

@author: saranya

This scrapes the radio stations to get the list of songs played for a specific city, at a specific date
"""

from __future__ import print_function
import bs4
import urllib
import csv

def songlyrics():
    
    lyrics = urllib.urlopen('http://b96.cbslocal.com/playlist/2016/11/20/')
    text = lyrics.read()
    soup = bs4.BeautifulSoup(text,'lxml')
    
    lyrics = soup.find_all('div',{'class':'playlist-item'})
    print(len(lyrics))
    with open('song_dataset_2016_11_20.csv','wb') as csvfile:
        data=csv.writer(csvfile)
        data.writerow(['Time','Song','Artist'])
        for lyric in lyrics:
            song=lyric['data-title']
            artist=lyric['data-artist']
            time=lyric.find('div',{'class':'time'}).string
            meridian = lyric.find('div',{'class':'meridian'}).string
            play_time=str(time)+' '+str(meridian)
            data.writerow([play_time, song,artist])
songlyrics()
