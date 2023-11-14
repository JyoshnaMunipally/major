import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv")str
st.title('Diabetes Checkup')
st.sidebar.header('Patients Data')
st.subheader('Training Data Stats')
st.write(df.describe())
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


def users_report():
    pregnancies = st.sidebar.slider('Pregnancies', 1, 15, 3)
    glucose = st.sidebar.slider('Glucose', 0, 220, 150)
    bp = st.sidebar.slider('Blood Pressure', 0, 142, 60)
    skin_thickness = st.sidebar.slider('Skin Thickness', 10, 110, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 75)
    bmi = st.sidebar.slider('BMI', 0, 69, 25)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.9, 0.47)
    age = st.sidebar.slider('Age', 0, 88, 39)

    users_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    report_data = pd.DataFrame(users_report_data, index=[0])
    return report_data


users_data = users_report()
st.subheader('Patients Data')
st.write(users_data)
rfor = RandomForestClassifier()
rfor.fit(x_train, y_train)
users_result = rfor.predict(users_data)
if users_result[0] == 0:
    color = 'blue'
else:
    color = 'red'

st.title('Visualised Patient Report')
st.header('Pregnancy Count Graph (Healthy v Not Healthy)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
ax2 = sns.scatterplot(x=users_data['Age'], y=users_data['Pregnancies'], s=150, color=color)
plt.xticks(np.arange(10, 110, 5))
plt.yticks(np.arange(0, 20, 4))
plt.title('0 - Healthy & 1 - Not Healthy')
st.pyplot(fig_preg)


st.header('Glucose Value Graph (Healthy v Not Healthy)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
ax4 = sns.scatterplot(x=users_data['Age'], y=users_data['Glucose'], s=150, color=color)
plt.xticks(np.arange(10, 110, 5))
plt.yticks(np.arange(0, 220, 10))
plt.title('0 - Healthy & 1 - Not Healthy')
st.pyplot(fig_glucose)
st.header('Blood Pressure Value Graph (Healthy v Not Healthy)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
ax6 = sns.scatterplot(x=users_data['Age'], y=users_data['BloodPressure'], s=150, color=color)
plt.xticks(np.arange(10, 110, 5))
plt.yticks(np.arange(0, 150, 10))
plt.title('0 - Healthy & 1 - Not Healthy')
st.pyplot(fig_bp)
st.header('Skin Thickness Value Graph (Healthy v Not Healthy)')
fig_str = plt.figure()
ax7 = sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues')
ax8 = sns.scatterplot(x=users_data['Age'], y=users_data['SkinThickness'], s=110, color=color)
plt.xticks(np.arange(10, 110, 5))
plt.yticks(np.arange(0, 110, 10))
plt.title('0 - Healthy & 1 - Not Healthy')
st.pyplot(fig_str)
st.header('Insulin Value Graph (Healthy v Not Healthy)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket')
ax10 = sns.scatterplot(x=users_data['Age'], y=users_data['Insulin'], s=150, color=color)
plt.xticks(np.arange(10, 110, 5))
plt.yticks(np.arange(0, 700, 50))
plt.title('0 - Healthy & 1 - Not Healthy')
st.pyplot(fig_i)
st.header('BMI Value Graph (Healthy v Not Healthy)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
ax12 = sns.scatterplot(x=users_data['Age'], y=users_data['BMI'], s=150, color=color)
plt.xticks(np.arange(10, 110, 5))
plt.yticks(np.arange(0, 80, 5))
plt.title('0 - Healthy & 1 - Not Healthy')
st.pyplot(fig_bmi)
st.header('DPF Value Graph (Healthy v Not Healthy)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x=users_data['Age'], y=users_data['DiabetesPedigreeFunction'], s=150, color=color)
plt.xticks(np.arange(10, 150, 5))
plt.yticks(np.arange(0, 3, 0.2))
plt.title('0 - Healthy & 1 - Not Healthy')
st.pyplot(fig_dpf)
st.subheader('Your Report')
output = ''
if users_result[0] == 0:
    output = 'You are not Diabetic'
else:
    output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy')
st.write(str(accuracy_score(y_test, rfor.predict(x_test)) * 100) + '%')
