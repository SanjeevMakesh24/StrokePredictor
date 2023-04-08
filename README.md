# StrokePredictor

Presentation: https://youtu.be/Huk_TAs_5t8

Visual Aid: https://drive.google.com/file/d/1TZzgD2AoPu2tLQY0Cha3_LnTOF3souwz/view?usp=sharing

Stroke Prediction Project
Overview
According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. This project aimed to build a predictive model that can identify patients who are at a higher risk of having a stroke based on their personal information, medical history, and lifestyle choices. The dataset was obtained from Kaggle and contained information about patients' gender, age, various diseases, smoking status, and more.

Dataset
The dataset we used is available on Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset. It contains 12 features, including the patient's age, gender, hypertension status, heart disease status, average glucose level, body mass index, and smoking status. The target variable is whether the patient had a stroke or not.

One of the challenges we faced with this dataset was that it was highly imbalanced, with only 4.9% of data points indicating that the patient had a stroke. To address this issue, we undersampled the data so that the model could be more accurate.

Model
We used logistic regression to build our predictive model. After cleaning and preprocessing the data, we split the dataset into training and testing sets with a 70:30 ratio. We then used gradient descent to optimize the model parameters.

During the project, we encountered an obstacle when trying to implement a front-end application through the use of React and connecting it to the Python application through an API created with Django. Unfortunately, both applications had technical problems that made them difficult to work with. Another obstacle we faced was that the initial model we built had an accuracy of 95%, but it was an overfit for the data. We also discovered that due to excessive testing resulting in a higher likelihood of the user not having a stroke, the model may have appeared more precise than it truly was.

The model was evaluated on a dataset of 1,473 data points, out of which 943 were True negative, 458 were False negative, 9 were False positive and 63 were True positive. Based on these numbers, the accuracy of the model was calculated to be 67.84%, the precision was 87.50%, the recall was 12.27%, and the F1 score was 21.46%. These numbers indicate that while the model has a relatively high precision (meaning that when it predicts a stroke, it is usually correct), it has a low recall (meaning that it misses a large number of actual strokes). This suggests that the model could benefit from further optimization to improve its sensitivity.

Future Improvements
While our team does not plan to continue working on this project, there are several potential avenues for future improvements. One idea is to connect with big companies such as iOS, Fitbit, and Android and utilize their data from health apps to do daily check-ups on patients. This means that people wouldn't have to go to annual general visits to the hospital, greatly increasing healthcare productivity by freeing up the schedule of doctors and providing care to those who need it the most. This idea aligns with the vision and mission of Axxess, which is to provide healthcare at the homefront, anytime, anywhere, and Neuro Rehab VR's XR Therapy Session, which aims to increase healthcare productivity.

Conclusion
In conclusion, our project aimed to predict whether a patient is likely to have a stroke based on their personal information, medical history, and lifestyle choices. We used logistic regression and undersampled the data to address the issue of imbalanced data. Although we encountered some obstacles during the project, we were able to build a model with an accuracy of 71.32%, which is within the range of industry-standard accuracy for logistic regression models. We hope that this project can serve as a starting point for future work on stroke prediction and healthcare productivity.





