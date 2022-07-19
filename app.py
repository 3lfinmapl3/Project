import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from turtle import color
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from flask_sqlalchemy import SQLAlchemy
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///Data.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class LiverDB(db.Model):
    sno = db.Column(db.Integer,primary_key=True)
    age = db.Column(db.Integer,nullable=False)
    gender = db.Column(db.Integer,nullable=False)
    total_bilirubin = db.Column(db.Integer,nullable=False)
    direct_bilirubin = db.Column(db.Integer,nullable=False)
    alkaline_phosphotase = db.Column(db.Integer,nullable=False)
    alamine_aminotransferase = db.Column(db.Integer,nullable=False)
    aspartate_aminotransferase = db.Column(db.Integer,nullable=False)
    total_protiens = db.Column(db.Integer,nullable=False)
    albumin = db.Column(db.Integer,nullable=False)
    albuminGlobulin = db.Column(db.Integer,nullable=False)

    def __repr__(self) -> str:
        return f"{self.sno}-{self.gender}"


class HeartDB(db.Model):
    sno = db.Column(db.Integer,primary_key = True)
    age = db.Column(db.Integer,nullable=False)
    gender = db.Column(db.Integer,nullable=False)
    cs = db.Column(db.Integer,nullable=False)
    cpd = db.Column(db.Integer,nullable=False)
    bpmeds = db.Column(db.Integer,nullable=False)
    pstrk = db.Column(db.Integer,nullable=False)
    phyp = db.Column(db.Integer,nullable=False)
    diab = db.Column(db.Integer,nullable=False)
    totChol = db.Column(db.Integer,nullable=False)
    sysBp = db.Column(db.Integer,nullable=False)
    diaBp = db.Column(db.Integer,nullable=False)
    bmi = db.Column(db.Integer,nullable=False)
    hrte = db.Column(db.Integer,nullable=False)
    gluc = db.Column(db.Integer,nullable=False)
    
    #1 means risk and 0 means norisk


    def __repr__(self) -> str:
        return f"{self.sno}-{self.gender}"
    

@app.route("/")
def hello_world():
    return render_template('Home.html')

@app.route("/Risk Predictions",methods=['GET','POST'])
def RiskPredictons():
    if request.method == 'GET':
        return render_template('Risk Predictions.html')
    else:
        test = pd.read_csv("./static/dataset/test_data.csv",error_bad_lines=False)
        train = pd.read_csv("./static/dataset/training_data.csv",error_bad_lines=False)
        train=train.drop('Unnamed: 133',axis=1)
        y_train=train.prognosis
        x_train=train.drop('prognosis',axis=1)
        y_test=test.prognosis
        x_test=test.drop('prognosis',axis=1)
        col=x_test.columns
        clf_rf = RandomForestClassifier(random_state=43)
        clr_rf = clf_rf.fit(x_train,y_train)
        pickle.dump(clr_rf,open('D:\Flask Python\static\model.pkl','wb'))
        input1 =  [str(x) for x in request.form.values()]
        b=[0]*132
        for x in range(0,132):
            for y in input1:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        b=b.reshape(1,132)
        prediction = clf_rf.predict(b)
        prediction=prediction[0]
       
        return render_template('result.html',predi=prediction,succ = 1)
@app.route("/Cardiovascular Analysis",methods=['GET','POST'])
def Cardiovascular():
    if request.method == 'GET':
        return render_template('alt_Cardiovascular.html')
    else:
        heartData = HeartDB(age = request.form['age'],gender = request.form['gender'],cs = request.form['cs'],cpd = request.form['cpd'],totChol = request.form['totChol'],bpmeds = request.form['bpm'],pstrk = request.form['pstr'],phyp=request.form['phyp'],diab = request.form['diab'],sysBp=request.form['sysBp'],diaBp=request.form['diaBp'], bmi = request.form['bmi'], hrte = request.form['hrte'], gluc = request.form['gluc'])
        db.session.add(heartData)
        db.session.commit()
        heart_dataset = pd.read_csv("D:\Flask Python\static\dataset\Heart2.csv")
        heart_dataset.dropna(inplace=True)
        heart_dataset = heart_dataset.drop(columns='education')
        X = heart_dataset.drop(columns='TenYearCHD',axis=1)
        Y = heart_dataset['TenYearCHD']
        X_train, X_test, Y_train, Y_test = train_test_split(X.values,Y,test_size=0.3,random_state=10)
        model = LogisticRegression(solver='liblinear',random_state = 0)
        model.fit(X_train,Y_train)
        y_pred = model.predict(X_test)
        sendData2 = (heartData.gender,heartData.age,heartData.cs,heartData.cpd,heartData.bpmeds,heartData.pstrk,heartData.phyp,heartData.diab,heartData.totChol,heartData.sysBp,heartData.diaBp,heartData.bmi,heartData.hrte,heartData.gluc)
        inputData = (heartData.gender,heartData.age,heartData.cs,heartData.cpd,heartData.bpmeds,heartData.pstrk,heartData.phyp,heartData.diab,heartData.totChol,heartData.sysBp,heartData.diaBp,heartData.bmi,heartData.hrte,heartData.gluc)
        inputData = np.asarray(inputData)
        inputData = inputData.reshape(1,-1)
        prediction = model.predict(inputData)
        smoke = "No Smoking Records found !"
        bpmeds = "No BP medicational suggestions found"
        diabetes = "No diabetic suggestions found"
        prevalent = "No prevalent stroke or hypertensive reactions recorded !"
        if(heartData.cs==1):
            smoke = "Smoking can act as a catalyst to bring you closer to a heart disease. Stop smoking today !"
            if(heartData.cpd > 1):
                smoke = smoke + "Even 1 cigeratte per day can kill you, avoid the number of cigerattes you smoke !"          
        if(heartData.bpmeds == 1):
            bpmeds = "If you find any discomfort taking your BP medicines, go see a doctor !"
        if(heartData.diab==1):
            diabetes = "If you take diabetic medicines, do not stop taking it on time."
        if(heartData.pstrk == 1 or heartData.phyp==1):
            prevalent = "If you are a prevalent stroke or hypertensive person, take your medications on time and schedule regular visits to the doctor."    

        if(heartData.age > 18):
            axis_range = np.arange(1)
            bar1 = plt.bar(axis_range,125,0.25,color="red")
            bar2 = plt.bar(axis_range + 0.25,heartData.totChol,0.25, color="green")
            bar3 = plt.bar(axis_range + 0.25 * 2, 200,0.25, color="orange")
            plt.xticks(axis_range+0.25,['Total Cholesterol'])
            plt.xlabel("Total Cholesterol")
            plt.ylabel("Range")
            plt.title("Your cholesterol range")
            plt.legend((bar1,bar2,bar3),('Lowest Normal Range','Your Range','Highest Normal Range'))
            plt.savefig('./static/Img/guestUserGraph/totChol.png')
            plt.clf()
            axis_range = np.arange(1)
            bar1 = plt.bar(axis_range,60,0.25,color="red")
            bar2 = plt.bar(axis_range + 0.25,heartData.hrte,0.25, color="green")
            bar3 = plt.bar(axis_range + 0.25 * 2, 100,0.25, color="orange")
            plt.xticks(axis_range+0.25,['Heart Rate'])
            plt.xlabel("Heart Rate")
            plt.ylabel("Range")
            plt.title("Your heart rate range")
            plt.legend((bar1,bar2,bar3),('Lowest Normal Rate','Your Rate','Highest Normal Rate'))
            plt.savefig('./static/Img/guestUserGraph/heartRate.png')
            plt.clf()


        if(heartData.age < 18):
            axis_range = np.arange(1)
            bar1 = plt.bar(axis_range,125,0.25,color="red")
            bar2 = plt.bar(axis_range + 0.25,heartData.totChol,0.25, color="green")
            bar3 = plt.bar(axis_range + 0.25 * 2, 170,0.25, color="orange")
            plt.xticks(axis_range+0.25,['Total Cholesterol'])
            plt.xlabel("Vital")
            plt.ylabel("Range")
            plt.title("Your vital range")
            plt.legend((bar1,bar2,bar3),('Lowest Normal Range','Your Range','Highest Normal Range'))
            plt.savefig('./static/Img/guestUserGraph/totChol.png')
            plt.clf()
            axis_range = np.arange(1)
            bar1 = plt.bar(axis_range,70,0.25,color="red")
            bar2 = plt.bar(axis_range + 0.25,heartData.hrte,0.25, color="green")
            bar3 = plt.bar(axis_range + 0.25 * 2, 100,0.25, color="orange")
            plt.xticks(axis_range+0.25,['Heart Rate'])
            plt.xlabel("Heart Rate")
            plt.ylabel("Range")
            plt.title("Your heart rate range")
            plt.legend((bar1,bar2,bar3),('Lowest Normal Rate','Your Rante','Highest Normal Rate'))
            plt.savefig('./static/Img/guestUserGraph/heartRate.png')
            plt.clf()     

        axis_range = np.arange(2)
        your_range = [heartData.sysBp,heartData.diaBp]
        highest_range = [120,80]
        lower_range = [10,5]
        bar1 = plt.bar(axis_range, lower_range,0.25,color="red")
        bar2 = plt.bar(axis_range + 0.25,your_range,0.25, color="green")
        bar3 = plt.bar(axis_range + 0.25 * 2, highest_range,0.25, color="orange")
        plt.xticks(axis_range+0.25,['Sys BP','Dia BP'])
        plt.xlabel("Blood Pressure")
        plt.ylabel("Range")
        plt.title("Your vital range")
        plt.legend((bar1,bar2,bar3),('Lowest Normal Range','Your Range','Highest Normal Range'))
        plt.savefig('./static/Img/guestUserGraph/bp.png')
        plt.clf()

        bar1 = plt.bar(np.arange(1),18.5,0.25,color="red")
        bar2 = plt.bar(np.arange(1) + 0.25,heartData.bmi,0.25, color="green")
        bar3 = plt.bar(np.arange(1) + 0.25 * 2,24.9,0.25, color="orange")
        plt.xticks(np.arange(1)+0.25,['BMI'])
        plt.xlabel("BMI")
        plt.ylabel("Range")
        plt.title("Your vital range")
        plt.legend((bar1,bar2,bar3),('Lowest Normal Range','Your Range','Highest Normal Range'))
        plt.savefig('./static/Img/guestUserGraph/bmi.png')
        plt.clf()

        bar1 = plt.bar(np.arange(1),50,0.25,color="red")
        bar2 = plt.bar(np.arange(1) + 0.25,heartData.gluc,0.25, color="green")
        bar3 = plt.bar(np.arange(1) + 0.25 * 2,139,0.25, color="orange")
        plt.xticks(np.arange(1)+0.25,['Glucose'])
        plt.xlabel("Glucose")
        plt.ylabel("Range")
        plt.title("Your vital range")
        plt.legend((bar1,bar2,bar3),('Lowest Normal Range','Your Range','Highest Normal Range'))
        plt.savefig('./static/Img/guestUserGraph/glucose.png')
        plt.clf()

        #Risk Prediction page = 0 CardioVascular = 1 Liver = 2 Kidney = 3
        return render_template('result.html',categ = 1,resulT = prediction[0],name =request.form['name'] ,sent = sendData2, history =prevalent, smoking = smoke, Diabetes=diabetes, bpmeds = bpmeds)

@app.route("/Liver Analysis",methods=['GET','POST'])
def Liver():
    if request.method == 'GET':
        return render_template('Liver Analysis.html')
    else :
        data = LiverDB(age = request.form['age'], gender = request.form['gender'] ,total_bilirubin = request.form['Total Bilirubin'],direct_bilirubin = request.form['Direct Bilirubin'],alkaline_phosphotase = request.form['Alkaline Phosphotase'],alamine_aminotransferase = request.form['Alamine Aminotransferase'],aspartate_aminotransferase = request.form['Aspartate Aminotransferase'],total_protiens = request.form['Total Protiens'],albumin = request.form['Albumin'],albuminGlobulin = request.form['AlbuminGlobulin'])
        db.session.add(data)
        db.session.commit()
        liver_dataset = pd.read_csv("D:\Flask Python\static\dataset\liver_patient.csv")
        liver_dataset['Gender'] = liver_dataset['Gender'].map({'Male': 1, 'Female': 2})
        liver_dataset.dropna(inplace=True)
        X = liver_dataset.drop(columns='Dataset', axis=1)
        Y = liver_dataset['Dataset']
        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.4, random_state=101)
        model1 = RandomForestClassifier(n_estimators = 100)
        model1.fit(X_train, Y_train)
        y_pred = model1.predict(X_test)
        input_data = (data.age,data.gender,data.total_bilirubin,data.direct_bilirubin,data.alkaline_phosphotase,data.alamine_aminotransferase,data.aspartate_aminotransferase,data.total_protiens,data.albumin,data.albuminGlobulin)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model1.predict(input_data_reshaped)
        
        
        cnt = 0
        if(data.gender == 1):
            gend = "Male"
        else:
            gend = "Female"
        #TB

        if(data.age > 18 and data.total_bilirubin > 1.2 and data.total_bilirubin < 5.0 ):
            tb= "Higher Than Normal"
        elif(data.age > 18 and data.total_bilirubin > 5.0 and data.total_bilirubin < 8.4):
            tb = "Moderately High"    
        elif(data.total_bilirubin <= 0):
            tb = "Low"
        elif(data.age < 18 and data.total_bilirubin > 1.0 and data.total_bilirubin < 4.0 ):
            tb = "Higher than normal"
        elif(data.age < 18 and data.total_bilirubin > 4.0 and data.total_bilirubin < 7.4 ):
            tb = "Moderately High"
        elif(data.age > 18 and data.total_bilirubin > 0 and data.total_bilirubin <= 1.2):
            tb = "Normal"
            cnt = cnt +1 
        elif(data.age < 18 and data.total_bilirubin > 0 and data.total_bilirubin <= 1.0):
            tb = "Normal"
            cnt = cnt +1
        else:
            tb = "Abnormally High"

        #DB
        
        if( data.direct_bilirubin >0.3 and data.direct_bilirubin < 0.5 ):
            dirb= "Higher Than Normal"
        elif( data.direct_bilirubin > 0 and data.direct_bilirubin <= 0.3 ):
            dirb= "Normal"
            cnt = cnt +1
        elif(data.direct_bilirubin >0.5 and data.direct_bilirubin < 0.8):
            dirb = "Moderately High"    
        elif(data.direct_bilirubin > 0.8 ):
            dirb = "Abnormally High"
        else:
            dirb = "Low"

        #ALP

        if( data.alkaline_phosphotase >147 and data.alkaline_phosphotase < 474 ):
            alp= "Higher Than Normal"
        elif(data.alkaline_phosphotase >474 and data.alkaline_phosphotase < 774):
            alp = "Moderately High" 
        elif(data.alkaline_phosphotase >=44 and data.alkaline_phosphotase <= 147):
            alp = "Normal" 
            cnt = cnt +1   
        elif(data.alkaline_phosphotase > 774 ):
            alp = "Abnormally High"
        else:
            alp = "Low"

        #ALT

        if((data.gender==1 and data.alamine_aminotransferase >= 29 and data.alamine_aminotransferase <= 33 ) or (data.gender==2 and data.alamine_aminotransferase >= 19 and data.alamine_aminotransferase <= 25 )):
            alt= "Normal"
            cnt = cnt +1
        elif((data.gender==1 and data.alamine_aminotransferase < 29) or (data.gender==2 and data.alamine_aminotransferase < 19)):
            alt = "Low" 
        elif((data.gender==1 and data.alamine_aminotransferase > 33 and data.alamine_aminotransferase < 300) or (data.gender==2 and data.alamine_aminotransferase > 25 and data.alamine_aminotransferase < 200)):
            alt = "Higher Than Normal"
        elif((data.gender==1 and data.alamine_aminotransferase > 300 and data.alamine_aminotransferase < 700) or (data.gender==2 and data.alamine_aminotransferase > 200 and data.alamine_aminotransferase < 400)):
            alt = "Moderately High"
        elif((data.gender==1 and data.alamine_aminotransferase > 700) or (data.gender==2 and data.alamine_aminotransferase > 400)):
            alt = "Abnormally High"             
        else:
            alt = "Computational Error"

       #AST

        if((data.gender==1 and data.aspartate_aminotransferase > 9 and data.aspartate_aminotransferase < 41 ) or (data.gender==2 and data.aspartate_aminotransferase > 8 and data.aspartate_aminotransferase < 33 )):
            ast= "Normal"
            cnt = cnt +1
        elif((data.gender==1 and data.aspartate_aminotransferase < 10) or (data.gender==2 and data.aspartate_aminotransferase < 9)):
            ast = "Low" 
        elif((data.gender==1 and data.aspartate_aminotransferase > 40 and data.aspartate_aminotransferase < 300) or (data.gender==2 and data.aspartate_aminotransferase > 32 and data.aspartate_aminotransferase < 292)):
            ast = "Higher Than Normal"
        elif((data.gender==1 and data.aspartate_aminotransferase > 300 and data.aspartate_aminotransferase < 800) or (data.gender==2 and data.aspartate_aminotransferase > 292 and data.aspartate_aminotransferase < 792)):
            ast = "Moderately High"
        elif((data.gender==1 and data.aspartate_aminotransferase > 800) or (data.gender==2 and data.aspartate_aminotransferase > 792)):
            ast = "Abnormally High"             
        else:
            ast = "Computational Error"

        #TP

        if( data.total_protiens > 8.3 and data.total_protiens < 9 ):
            tp= "Higher Than Normal"
        elif(data.total_protiens >9 and data.total_protiens < 15):
            tp = "Moderately High" 
        elif(data.total_protiens >=6.0 and data.total_protiens <= 8.3):
            tp = "Normal"    
            cnt = cnt +1
        elif(data.total_protiens > 15 ):
            tp = "Abnormally High"
        else:
            tp = "Low"

        #ALB

        if( data.albumin > 5.6 and data.albumin < 7 ):
            al= "Higher Than Normal"
        elif(data.albumin >7 and data.albumin < 15):
            al = "Moderately High" 
        elif(data.albumin >=3.5 and data.albumin <= 5.6):
            al = "Normal"  
            cnt = cnt +1  
        elif(data.albumin > 15 ):
            al = "Abnormally High"
        else:
            al = "Low"

        #ALBGLB

        if( data.albuminGlobulin > 1 and data.albuminGlobulin < 2.5 ):
            algl= "Higher Than Normal"
        elif(data.albuminGlobulin >2.5 and data.albuminGlobulin < 5):
            algl = "Moderately High" 
        elif(data.albuminGlobulin ==1):
            algl = "Normal" 
            cnt = cnt +1   
        elif(data.albuminGlobulin > 5 ):
            algl = "Abnormally High"
        else:
            algl = "Low"
        
      
        
        #For accuracy
        accuracy = metrics.accuracy_score(Y_test,y_pred) * 10

        if(cnt >= 5):
            result = "Your Health Score is {} / 8, you seem fit but you need to worry about the deteriorating factors".format(cnt)
        elif(cnt == 8):
            result = "Your health score is 8 / 8, your liver is functioning fine !"   
        else:
            result = "Your healt score is {} / 8 and your liver is at risk. Schedule a meet with your doctor for emergency results !".format(cnt) 

        axis_x = 3
        axis_range = np.arange(axis_x)
        if(data.gender==1):
            highest_range = [147.0,33.0,40.0]
            lower_range = [44.0,29.0,10.0]
        else:
            highest_range = [147.0,25.0,32.0]
            lower_range = [44.0,19.0,9.0]
       
        your_range = [data.alkaline_phosphotase,data.alamine_aminotransferase,data.aspartate_aminotransferase]
        bar1 = plt.bar(axis_range, lower_range,0.25,color="red")
        bar2 = plt.bar(axis_range + 0.25,your_range,0.25, color="green")
        bar3 = plt.bar(axis_range + 0.25 * 2, highest_range,0.25, color="orange")
        plt.xticks(axis_range+0.25,['ALP','ALT','AST'])
        plt.xlabel("Vitals")
        plt.ylabel("Range")
        plt.title("Your vital range")
        plt.legend((bar1,bar2,bar3),('Lowest Normal Range','Your Range','Highest Normal Range'))
        plt.savefig('D:\Flask Python\static\Img\guestUserGraph\guestUserReport1.png')
        plt.clf()

        axis_x = 5
        axis_range = np.arange(axis_x)
        highest_range = [1.2,0.3,8.3,5.6,1.0]
        lower_range = [0.1,0.1,6.0,3.5,1.0]
        your_range = [data.total_bilirubin, data.direct_bilirubin,data.total_protiens,data.albumin,data.albuminGlobulin]
        bar1 = plt.bar(axis_range, lower_range,0.25,color="red")
        bar2 = plt.bar(axis_range + 0.25,your_range,0.25, color="green")
        bar3 = plt.bar(axis_range + 0.25 * 2, highest_range,0.25, color="orange")
        plt.xticks(axis_range+0.25,['TB','DB','TP','ALB','ALB:GLB'])
        plt.xlabel("Vitals")
        plt.ylabel("Range")
        plt.title("Your vital range")
        plt.legend((bar1,bar2,bar3),('Lowest Normal Range','Your Range','Highest Normal Range'))
        plt.savefig('./static/Img/guestUserGraph/guestUserReport2.png')
        plt.clf()
        
        
    return render_template('result.html',categ = 2,resulT = prediction[0],  resultValue = result, userName = "Guest_User", age = data.age, gender = gend, totb= tb, dirtb = dirb, ALP = alp, ALT=alt, AST = ast, TP =tp, AL = al, ALGL = algl, name = request.form['name'])
       

if __name__ == "__main__":
    app.run(debug=True, port=8000)