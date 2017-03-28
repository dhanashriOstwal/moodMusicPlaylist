#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:08:55 2016

@author: dhanashri
"""

import time
import random
import csv
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts
from sklearn.externals import joblib
from nltk_preprocessor import NLTKPreprocessor 
import re
import os

import urllib2
import json
import operator
#import NLTKPreprocessor

def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg
    
def readCSV(filename):
    csvRows = []
    with open(filename, 'rU') as csvfile:
        rows = csv.reader(csvfile)
        print(rows)
        for row in rows:
            if not row[1] == 'OTHER_FLOW':
                row[0] = unicode(row[0], errors='replace')
                csvRows.append(row)
    csvfile.close()
    return csvRows


@timeit
def build_and_evaluate(X, y, classifier=SGDClassifier(loss='log'), outpath=None, verbose=True):
    """
    Builds a classifer for the given list of documents and targets in two
    stages: the first does a train/test split and prints a classifier report,
    the second rebuilds the model on the entire corpus and returns it for
    operationalization.
    X: a list or iterable of raw strings, each representing a document.
    y: a list or iterable of labels, which will be label encoded.
    Can specify the classifier to build with: if a class is specified then
    this will build the model with the Scikit-Learn defaults, if an instance
    is given, then it will be used directly in the build pipeline.
    If outpath is given, this function will write the model as a pickle.
    If verbose, this function will print out information to the command line.
    """
    @timeit
    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()
            
        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
            ('classifier', classifier),
        ])
        model.fit(X, y)
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y)
    
    # Begin evaluation
    if verbose: print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model, secs = build(classifier, X_train, y_train)

    if verbose: print("Evaluation model fit in {:0.3f} seconds".format(secs))
    if verbose: print("Classification Report:\n")

    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))

    if verbose: print("Building complete model and saving ...")
    model, secs = build(classifier, X, y)
    model.labels_ = labels

    if verbose: print("Complete model fit in {:0.3f} seconds".format(secs))

    if outpath:
        # Need to dump NLTKPreprocessor object too, else will get error while
        # loading the model
        #with open(outpath, 'wb') as f:
        joblib.dump(model,outpath)
        print("Model written out to {}".format(outpath))

    return model

def predict_music(model, musicData): 
    happyAccuracy = []
    sadAccuracy = []
    class_prediction = [] 
    for lyr in musicData: 
        try:         
            pred = model.predict(lyr[2])
        except UnicodeDecodeError:
            str1 = ''.join(lyr[2])
            lyr[2] = ((re.sub(r'[^\x00-\x7F]+',' ', str1)))
            pred = model.predict([lyr[2]])
        probs = model.predict_proba(lyr[2])
        print("Model: ", lyr[0])
        print("Prediction: ", model.labels_.inverse_transform(pred[0]))
        class_prediction.append([lyr[0],lyr[1],model.labels_.inverse_transform(pred[0])])
        probs = model.predict_proba(lyr[2])
        best_n = np.argsort(probs, axis=1)[-3:]
        cnt = 0
        
        result = {}
        for best in reversed(best_n[0]):
            if cnt < 3:
                #print best
                #print text_clf.classes_[best]
                #result[text_clf.classes_[best]] = probs[0][best]
                result[model.classes_[best]] = "{0:.0f}%".format(probs[0][best] * 100)
                cnt = cnt + 1

        #print "The top 3 classifications are:"
        #print(model.labels_.inverse_transform(upses))
        for c,v in result.iteritems():
            print(model.labels_.inverse_transform(c)," - ",v)
            if model.labels_.inverse_transform(c) == "Happy":
                removePercent = v.replace("%","")
                happyAccuracy.append(float(removePercent))
            else:
                removePercent = v.replace("%","")
                sadAccuracy.append(float(removePercent))
        print('\n')

    happyPercent = sum(happyAccuracy)/len(happyAccuracy)
    sadPercent = float(sum(sadAccuracy)/len(sadAccuracy))
    return happyPercent,sadPercent,class_prediction

def weather_api(city,date):
    location={'dallas':'TX','chicago':'IL','newyork':'NY'}
    if city in location :
        state = location[city]
    
    date=date.replace('_','')
#    print(date)
    
    url='http://api.wunderground.com/api/c1e07de82ce6f3bb/history_'+ date + '/q/'+state+'/'+city+'.json'
    f = urllib2.urlopen(url)
    json_string = f.read()
    parsed_json = json.loads(json_string)    
    temp_history = parsed_json['history']['observations']
    cond={}
    for temp in temp_history:
    
        if temp['conds'] in cond:
            cond[temp['conds']]+=1
        else:
            cond[temp['conds']]=1
    weather_cond=max(cond.iteritems(),key=operator.itemgetter(1))[0]
    return weather_cond
            
        
        #t=temp['tempi']
        #sum_temp+=float(t)
        #avg_temp=sum_temp/len(temp_history)
    #print("Average temp for {} on {} is {}".format(loc[1],date,avg_temp))
    f.close()

percentValuesHappy = []
percentValuesSad = []
PATH = "model.sav"

# Read the csv dataset
csvRows = readCSV("LyricsClassDataset.csv")    
        
# Separate text and categories
shuffled_data = random.sample(csvRows, len(csvRows))
data,cats = zip(*shuffled_data)

#Generating the model only once using the training dataset
#model = build_and_evaluate(data,cats, outpath=PATH)
filenames=[]
script_dir = os.path.dirname(__file__)
rel_path = "Classified Test Data/"
direct = os.path.join(script_dir, rel_path)
    #direct= 'G:/final_sem_datascience/final_project/pokemon_go/data/'
for roots,dirs,files in os.walk(direct):    
    for name in files:
        filenames.append(name)

for filename in filenames:
    city = filename.split('_')[1]
    date = filename[-10:] 
    file_for_prediction = filename.split('_',1)
    print(file_for_prediction[1])
    musicData = []
    with open('Classified Test Data/{}.csv'.format(filename),'r') as fname:
        lines = csv.reader(fname)
        for line in lines:
            lyrics = []
            title = []
            artist = []
            lyrics.append(line[3])
            title.append(line[2])
            artist.append(line[1])
            musicData.append([title,artist,lyrics])
            
        model = joblib.load("model.sav")
        cnt = 1
        
        #Passing already generated model and the test data to predict_data
        happyPercent,sadPercent,class_prediction = predict_music(model, musicData)
#        print(happyPercent,sadPercent) 
        classification = [date,happyPercent,sadPercent]
        
        #Writing to a csv file. (For Bar Graph)
        with open('Graphs/{}.csv'.format(city),'ab') as csvfile:
            song_details = csv.writer(csvfile)
            song_details.writerow(classification)
        csvfile.close()
        
        happy_data=[]
        sad_data=[]
        count_classify_dict = {}
        for i in range(len(class_prediction)):
            if class_prediction[i][2] == 'Happy':
                happy_data.append([(', '.join(str(s) for s in class_prediction[i][0])), (', '.join(str(s) for s in class_prediction[i][1]))])
            else:
                sad_data.append([(', '.join(str(s) for s in class_prediction[i][0])), (', '.join(str(s) for s in class_prediction[i][1]))])
        
        #List of recommended song list        
        weather_condition = weather_api(city,date)
        with open('Graphs/{}.csv'.format(file_for_prediction[1]),'ab') as csvF:
            details = csv.writer(csvF)
            details.writerow([weather_condition])
            if len(happy_data) > len(sad_data):
                details.writerows(happy_data)
            else:
                details.writerows(sad_data)