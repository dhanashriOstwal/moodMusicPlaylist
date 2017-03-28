# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 16:09:05 2016

@author: Dhanashri
"""

import nltk
import webScrape as wS
import csv
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

#Stop word removal
def remove_stop_words(text):
    keyword = ' '
    stop = set(nltk.corpus.stopwords.words('english'))
    for i in text.lower().split():
        if i not in stop:
            try:
                stemmedVar = ps.stem(i)
            except UnicodeDecodeError:
                continue
            keyword += ' ' + stemmedVar
    return keyword

# Sentiment Analysis using TextBlob Naive Bayes
def classification_of_lyrics(lyrics):
    count_pos = 0
    count_neg = 0
    lyrics_without_stopwords = remove_stop_words(lyrics)
    print(lyrics_without_stopwords)
    text = TextBlob(lyrics_without_stopwords,analyzer = NaiveBayesAnalyzer())
    if text.sentiment.classification == 'pos':
        count_pos += 1
    else:
        count_neg += 1
    if count_pos > count_neg:
        return "Happy"
    else:
        return "Sad"

cnt = 0
count = 1
rec_cnt = 1
songDetails = []
with open('ArtistList.csv','r') as f:
    lines = csv.reader(f)
    #skip header line
    for line in lines:
        print(rec_cnt)
        rec_cnt += 1
        if not cnt == 0:
            artist = line[2]
            song_title = line[3]
            #Call to get_lyrics from WebScrape.py
            lyr = wS.get_lyrics(artist,song_title)
            if lyr > 0:
                song_classify = classification_of_lyrics(lyr)
                songDetails = [artist,song_title,song_classify,lyr]
                #Writing in a csv file - artist, song title, classification of song and lyrics
                with open('ClassifiedSongDataset.csv', 'ab') as csvfile:
                    song_details = csv.writer(csvfile)
                    song_details.writerow(songDetails)
                csvfile.close()
                print("Song ",count)
                count += 1
                songDetails = []

        else:
            cnt = 1
    f.close()
    