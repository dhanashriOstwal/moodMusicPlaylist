from flask import Flask, render_template,request
import os
import csv


app = Flask(__name__)

HOST = str(os.getenv('VCAP_APP_HOST', 'localhost'))
PORT = int(os.getenv('VCAP_APP_PORT', '3120'))

global dt


@app.route ('/', methods=['POST', 'GET'])
def main():
     return render_template('first.html')

#Graph to visualize confidence level of 3 cities
@app.route('/chart', methods=['POST'])
def chart():
    global dt
    dt = request.form['dt']
    print (dt)
    labels = ["dallas","chicago","newyork"]
    happy={}
    sad={}
    #Read the contents of CSV file to categorize as happy and sad
    for city in labels:
        fname= "Graphs/"+city+".csv"
        with open(fname,'r') as csvfile:
            reader=csv.reader(csvfile)
            for row in reader:
                if row[0]==dt:
                    happy[city]=row[1]
                    sad[city]=row[2]
    
    
    return render_template('chart.html',date=dt, graph1=happy, graph2=sad)

#Recommendation of songs based on weather by reading CSV file    
@app.route('/recommendation', methods=['POST'])
def recommendation():
    location=request.form['loc']
    filename='Graphs/'+location+'_dataset_'+dt+'.csv'
    album_list=[]
    with open(filename,'r') as csvf:
        rd=csv.reader(csvf)
        weather=next((rd))        
        for line in rd:
            album_list.append(('  -  '.join(str(s) for s in line)))
    return render_template('recommended_songs.html',alb_list=album_list,date=dt,loc=location,weather=weather[0])
            


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)

