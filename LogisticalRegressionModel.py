import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
from imblearn.under_sampling import RandomUnderSampler

#calculated through gradient decent
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
#print(X.info())

#print(df[df['stroke']==1].count())



# Split the data into training and testing sets
X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.3, random_state=42)

##oversampling
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

pickle.dump(lr, open('Model.pkl', 'wb'))
