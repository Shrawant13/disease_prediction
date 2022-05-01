import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

app = Flask(__name__)

model1 = pickle.load(open("model1.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    select = request.form.getlist('selected')

    var=['Itching',
    'Skin rash',
    'Nodal skin eruptions',
    'Continuous sneezing',
    'Shivering',
    'Chills',
    'Joint pain',
    'Stomach pain',
    'Acidity',
    'Ulcers on tongue',
    'Muscle wasting',
    'Vomiting',
    'Burning micturition',
    'Spotting  urination',
    'Fatigue',
    'Weight gain',
    'Anxiety',
    'Cold hands and feets',
    'Mood swings',
    'Weight loss',
    'Restlessness',
    'Lethargy',
    'Patches in throat',
    'Irregular sugar level',
    'Cough',
    'High fever',
    'Sunken eyes',
    'Breathlessness',
    'Sweating',
    'Dehydration',
    'Indigestion',
    'Headache',
    'Yellowish skin',
    'Dark urine',
    'Nausea',
    'Loss of appetite',
    'Pain behind the eyes',
    'Back pain',
    'Constipation',
    'Abdominal pain',
    'Diarrhoea',
    'Mild fever',
    'Yellow urine',
    'Yellowing of eyes',
    'Acute liver failure',
    'Fluid overload',
    'Swelling of stomach',
    'Swelled lymph nodes',
    'Malaise',
    'Blurred and distorted vision',
    'Phlegm',
    'Throat irritation',
    'Redness of eyes',
    'Sinus pressure',
    'Runny nose',
    'Congestion',
    'Chest pain',
    'Weakness in limbs',
    'Fast heart rate',
    'Pain during bowel movements',
    'Pain in anal region',
    'Bloody stool',
    'Irritation in anus',
    'Neck pain',
    'Dizziness',
    'Cramps',
    'Bruising',
    'Obesity',
    'Swollen legs',
    'Swollen blood vessels',
    'Puffy face and eyes',
    'Enlarged thyroid',
    'Brittle nails',
    'Swollen extremeties',
    'Excessive hunger',
    'Extra marital contacts',
    'Drying and tingling lips',
    'Slurred speech',
    'Knee pain',
    'Hip joint pain',
    'Muscle weakness',
    'Stiff neck',
    'Swelling joints',
    'Movement stiffness',
    'Spinning movements',
    'Loss of balance',
    'Unsteadiness',
    'Weakness of one body side',
    'Loss of smell',
    'Bladder discomfort',
    'Foul smell of urine',
    'Continuous feel of urine',
    'Passage of gases',
    'Internal itching',
    'Toxic look (typhos)',
    'Depression',
    'Irritability',
    'Muscle pain',
    'Altered sensorium',
    'Red spots over body',
    'Belly pain',
    'Abnormal menstruation',
    'Dischromic  patches',
    'Watering from eyes',
    'Increased appetite',
    'Polyuria',
    'Family history',
    'Mucoid sputum',
    'Rusty sputum',
    'Lack of concentration',
    'Visual disturbances',
    'Receiving blood transfusion',
    'Receiving unsterile injections',
    'Coma',
    'Stomach bleeding',
    'Distention of abdomen',
    'History of alcohol consumption',
    'Fluid overload.1',
    'Blood in sputum',
    'Prominent veins on calf',
    'Palpitations',
    'Painful walking',
    'Pus filled pimples',
    'Blackheads',
    'Scurring',
    'Skin peeling',
    'Silver like dusting',
    'Small dents in nails',
    'Inflammatory nails',
    'Blister',
    'Red sore around nose',
    'Yellow crust ooze']

    for x in var:
        for x in select:
            for i in range(len(var)):
                if var[i] == x:
                    var[i] = 1
     
    for x in var:
        for i in range(len(var)):
            if var[i] != 1:
                var[i] = 0   

    asd=[var]
    train=pd.read_csv('Training.csv')
    test=pd.read_csv('Testing.csv')
    disease = pd.DataFrame(test['Prognosis'])
    disease=disease.iloc[:132]
    train['Prognosis']=train['Prognosis'].astype('category')
    test['Prognosis']=test['Prognosis'].astype('category')
    from sklearn.model_selection import train_test_split
    train_target=train["Prognosis"]
    train1=train.drop('Prognosis', axis=1)
    train2,val2,train_target2,val_target2 = train_test_split(train1, train_target, test_size=0.3, random_state=42)  
    model1 = RandomForestClassifier(n_estimators=500,max_features=30,max_depth=2)
    model1.fit(train1, train_target)
    prediction = model1.predict(asd)

    return render_template('index.html', prediction_text='The predicted disease for the given symptoms is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)