import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import ttk

import pickle


import pandas as pd

from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score


########################################################################################################################

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

#drops rows with null values
df = df.dropna()

df['gender'].replace(['Female','Male', 'Other'], [0, 1, 2], inplace = True)
df['ever_married'].replace(['No', 'Yes'], [0, 1], inplace = True)
df['work_type'].replace(['Private', 'Self-employed', 'Govt_job', 'Never_worked', 'children'], [1, 2, 3, 4, 5], inplace= True)
df['Residence_type'].replace(['Rural', 'Urban'], [0, 1], inplace = True)
df['smoking_status'].replace(['never smoked', 'Unknown', 'formerly smoked', 'smokes'], [1, 2, 3, 4], inplace = True)


# Split the data into features (X --> input values) and target 
df.drop(columns=['id'], inplace = True)
X = df.drop(columns=['stroke'])
target = df['stroke']


# Split the data into training and testing sets
X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.3, random_state=42)

##oversampling - accuracy 76 - but not working properly
#oversampler = SMOTE(random_state=42)
#X_train, target_train = oversampler.fit_resample(X_train, target_train)

#undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train, target_train = undersampler.fit_resample(X_train, target_train)


# Create a logistic regression object
lr = LogisticRegression()

#trains the logistic regression model on the training data (X_train, target_train) 
#that we split earlier using train_test_split()
#The fit() method is used to train a machine learning model on the given data.
lr.fit(X_train, target_train)

# Evaluate the model on the testing set
accuracy = lr.score(X_test, target_test)
print("Accuracy:", accuracy)

# Use the trained model to make predictions on the testing set
predictions = lr.predict(X_test)

# Calculate the precision of the predictions
precision = precision_score(target_test, predictions)

# Print the precision
print("Precision:", precision)

pickle.dump(lr, open('Model.pkl', 'wb'))

########################################################################################################################

pickled_model = pickle.load(open('Model.pkl', 'rb'))

# Create a GUI window
window = tk.Tk()
window.title("Stroke Prediction")

tk.Label(window, font = ('Helvetica', 12))


# Create input fields and labels
tk.Label(window, text="Gender (Female = 0, Male = 1, Other = 2)", font = ("Georgia", 10)).grid(row=0)
tk.Label(window, text="Age", font = ("Georgia", 10)).grid(row=1)
tk.Label(window, text="Hypertension (No = 0, Yes = 1)", font = ("Georgia", 10)).grid(row=2)
tk.Label(window, text="Heart Disease (No = 0, Yes = 1)", font = ("Georgia", 10)).grid(row=3)
tk.Label(window, text="Ever Married (No = 0, Yes = 1)", font = ("Georgia", 10)).grid(row=4)
tk.Label(window, text="Work Type (Private = 1, Self-employed = 2, Govt_job = 3, Never_worked = 4, Children = 5)", font = ("Georgia", 10)).grid(row=5)
tk.Label(window, text="Residence Type (Rural = 0, Urban = 1)", font = ("Georgia", 10)).grid(row=6)
tk.Label(window, text="Average Glucose Level", font = ("Georgia", 10)).grid(row=7)
tk.Label(window, text="Weight (in lbs)", font = ("Georgia", 10)).grid(row=8)
tk.Label(window, text="Height (in inches)", font = ("Georgia", 10)).grid(row=9)
tk.Label(window, text="Smoking Status (Never smoked = 1, Unknown = 2, Formerly smoked = 3, Smokes = 4)", font = ("Georgia", 10)).grid(row=10)

e1 = ttk.Combobox(window, values=[0,1])
e2 = tk.Entry(window)
e3 = ttk.Combobox(window, values=[0, 1])
e4 = ttk.Combobox(window, values=[0, 1])
e5 = ttk.Combobox(window, values=[0, 1])
e6 = ttk.Combobox(window, values=[1,2,3,4,5])
e7 = ttk.Combobox(window, values=[0, 1])
e8 = tk.Entry(window)
e9 = tk.Entry(window)
e10 = tk.Entry(window)
e11 = ttk.Combobox(window, values=[1,2,3,4])


#print(e11)


e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)
e7.grid(row=6, column=1)
e8.grid(row=7, column=1)
e9.grid(row=8, column=1)
e10.grid(row=9, column=1)
e11.grid(row=10, column=1)

# Function to predict stroke based on input values
def predict_stroke():
    # Get input values from GUI entry fields
    gender = int(e1.get())
    age = int(e2.get())
    hypertension = int(e3.get())
    heart_disease = int(e4.get())
    ever_married = int(e5.get())
    work_type = int(e6.get())
    residence_type = int(e7.get())
    avg_glucose_level = float(e8.get())
    weight = float(e9.get())
    height = float(e10.get())
    smoking_status = int(e11.get())

    bmi = (weight / (height * height)) * 703
    # Create new data frame with input values
    new_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Use logistic regression model to make prediction
    pred = pickled_model.predict(new_data)
    prob = pickled_model.predict_proba(new_data)
    

    # Display result to user
    if pred[0] == 1:
        result_label.config(text="Prediction: Stroke\nProbability: {:.4f}%".format(prob[0][1]*100))
    else:
        result_label.config(text="Prediction: No Stroke\nProbability: {:.4f}%".format(prob[0][0]*100))

    

# Create "Predict" button and result label
predict_button = tk.Button(window, text="Predict", command=predict_stroke)
predict_button.grid(row=11, column=0, pady=10)

result_label = tk.Label(window, text="")
result_label.grid(row=11, column=1)


window.mainloop()

target_predict = pickled_model.predict(X_test)
cm = confusion_matrix(target_test,target_predict)

print(cm)