import os

import numpy as np
import pandas as pd
from scipy.stats import mode
from flask import Flask, request, render_template, jsonify, make_response
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv
import json


app = Flask(__name__, template_folder='template')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_disease():
    print(request.data)
    data =json.loads(request.data)
    symptoms=data["symptoms"]


    data_dict={'symptom_index': {'Itching': 0, 'Skin Rash': 1, 'Nodal Skin Eruptions': 2, 'Continuous Sneezing': 3, 'Shivering': 4, 'Chills': 5, 'Joint Pain': 6, 'Stomach Pain': 7, 'Acidity': 8, 'Ulcers On Tongue': 9, 'Muscle Wasting': 10, 'Vomiting': 11, 'Burning Micturition': 12, 'Spotting  urination': 13, 'Fatigue': 14, 'Weight Gain': 15, 'Anxiety': 16, 'Cold Hands And Feets': 17, 'Mood Swings': 18, 'Weight Loss': 19, 'Restlessness': 20, 'Lethargy': 21, 'Patches In Throat': 22, 'Irregular Sugar Level': 23, 'Cough': 24, 'High Fever': 25, 'Sunken Eyes': 26, 'Breathlessness': 27, 'Sweating': 28, 'Dehydration': 29, 'Indigestion': 30, 'Headache': 31, 'Yellowish Skin': 32, 'Dark Urine': 33, 'Nausea': 34, 'Loss Of Appetite': 35, 'Pain Behind The Eyes': 36, 'Back Pain': 37, 'Constipation': 38, 'Abdominal Pain': 39, 'Diarrhoea': 40, 'Mild Fever': 41, 'Yellow Urine': 42, 'Yellowing Of Eyes': 43, 'Acute Liver Failure': 44, 'Fluid Overload': 45, 'Swelling Of Stomach': 46, 'Swelled Lymph Nodes': 47, 'Malaise': 48, 'Blurred And Distorted Vision': 49, 'Phlegm': 50, 'Throat Irritation': 51, 'Redness Of Eyes': 52, 'Sinus Pressure': 53, 'Runny Nose': 54, 'Congestion': 55, 'Chest Pain': 56, 'Weakness In Limbs': 57, 'Fast Heart Rate': 58, 'Pain During Bowel Movements': 59, 'Pain In Anal Region': 60, 'Bloody Stool': 61, 'Irritation In Anus': 62, 'Neck Pain': 63, 'Dizziness': 64, 'Cramps': 65, 'Bruising': 66, 'Obesity': 67, 'Swollen Legs': 68, 'Swollen Blood Vessels': 69, 'Puffy Face And Eyes': 70, 'Enlarged Thyroid': 71, 'Brittle Nails': 72, 'Swollen Extremeties': 73, 'Excessive Hunger': 74, 'Extra Marital Contacts': 75, 'Drying And Tingling Lips': 76, 'Slurred Speech': 77, 'Knee Pain': 78, 'Hip Joint Pain': 79, 'Muscle Weakness': 80, 'Stiff Neck': 81, 'Swelling Joints': 82, 'Movement Stiffness': 83, 'Spinning Movements': 84, 'Loss Of Balance': 85, 'Unsteadiness': 86, 'Weakness Of One Body Side': 87, 'Loss Of Smell': 88, 'Bladder Discomfort': 89, 'Foul Smell Of urine': 90, 'Continuous Feel Of Urine': 91, 'Passage Of Gases': 92, 'Internal Itching': 93, 'Toxic Look (typhos)': 94, 'Depression': 95, 'Irritability': 96, 'Muscle Pain': 97, 'Altered Sensorium': 98, 'Red Spots Over Body': 99, 'Belly Pain': 100, 'Abnormal Menstruation': 101, 'Dischromic  Patches': 102, 'Watering From Eyes': 103, 'Increased Appetite': 104, 'Polyuria': 105, 'Family History': 106, 'Mucoid Sputum': 107, 'Rusty Sputum': 108, 'Lack Of Concentration': 109, 'Visual Disturbances': 110, 'Receiving Blood Transfusion': 111, 'Receiving Unsterile Injections': 112, 'Coma': 113, 'Stomach Bleeding': 114, 'Distention Of Abdomen': 115, 'History Of Alcohol Consumption': 116, 'Fluid Overload.1': 117, 'Blood In Sputum': 118, 'Prominent Veins On Calf': 119, 'Palpitations': 120, 'Painful Walking': 121, 'Pus Filled Pimples': 122, 'Blackheads': 123, 'Scurring': 124, 'Skin Peeling': 125, 'Silver Like Dusting': 126, 'Small Dents In Nails': 127, 'Inflammatory Nails': 128, 'Blister': 129, 'Red Sore Around Nose': 130, 'Yellow Crust Ooze': 131}, 'predictions_classes':['(vertigo) Paroymsal  Positional Vertigo', 'Fever', 'Acne',
       'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
       'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
       'Common Cold', 'Dengue', 'Diabetes ',
       'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
       'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
       'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
       'Osteoarthristis', 'Paralysis (brain hemorrhage)',
       'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
       'Typhoid', 'Urinary tract infection', 'Varicose veins',
       'hepatitis A']}
    symptoms = symptoms.split(",")

    # Training the models on whole data
    DATA_PATH = "Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis=1)


    # Checking whether the dataset is balanced or not
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24)

    final_svm_model = SVC()
    final_nb_model = GaussianNB()
    final_rf_model = RandomForestClassifier(random_state=18)
    final_svm_model.fit(X_train, y_train)
    final_nb_model.fit(X_train, y_train)
    final_rf_model.fit(X_train, y_train)
    final_svm_model.predict(X_test)
    final_nb_model.predict(X_test)
    final_rf_model.predict(X_test)






    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:

        symptom = " ".join([i.capitalize() for i in symptom.split(" ")])
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    print(input_data)

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    medicine='Medicine not available'
    # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    with open("diseases_data.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader:
            print(row[2])
            if(row[0]==final_prediction):
                medicine=row[2]
                break

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction,
        "medicine":medicine
    }
    return make_response( {"rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction,
        "medicine":medicine});

if __name__=="__main__":

    app.run(debug=True,port=int(os.environ.get('PORT', 8080)))










