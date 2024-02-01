
# coding: utf-8
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load the dataset
df_1 = pd.read_csv("Telco-Customer-Churn.csv")

@app.route("/")
def loadPage():
    return render_template("home.html", query="")

@app.route("/", methods=['POST'])
def predict():
    # Load the trained model
    model = pickle.load(open("model.sav", "rb"))
    
    # Get form inputs
    input_query = [request.form[f'query{i+1}'] for i in range(19)]

    # Create DataFrame from form inputs
    new_df = pd.DataFrame([input_query], 
                          columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                   'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                   'PaymentMethod', 'tenure'])

    # Concatenate new_df with df_1
    df_2 = pd.concat([df_1, new_df], ignore_index=True) 
    
    # Dummy encoding
    new_df__dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                            'Contract', 'PaperlessBilling', 'PaymentMethod']])
    
    # Predict churn
    single = model.predict(new_df__dummies.tail(1))
    probability = model.predict_proba(new_df__dummies.tail(1))[:,1]
    
    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = f"Confidence: {probability * 100}"
    else:
        o1 = "This customer is likely to continue!!"
        o2 = f"Confidence: {probability * 100}"
        
    return render_template("home.html", output1=o1, output2=o2, 
                           query1=request.form['query1'], 
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'], 
                           query6=request.form['query6'], 
                           query7=request.form['query7'], 
                           query8=request.form['query8'], 
                           query9=request.form['query9'], 
                           query10=request.form['query10'], 
                           query11=request.form['query11'], 
                           query12=request.form['query12'], 
                           query13=request.form['query13'], 
                           query14=request.form['query14'], 
                           query15=request.form['query15'], 
                           query16=request.form['query16'], 
                           query17=request.form['query17'],
                           query18=request.form['query18'], 
                           query19=request.form['query19'])

if __name__ == "__main__":
    app.run()
