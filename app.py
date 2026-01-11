import pandas as pd
import numpy as np
import pickle
import streamlit as st



lr = pickle.load(open('lr.pkl','rb'))   # rb = read binary
dt = pickle.load(open('dtc.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))

model = st.sidebar.selectbox('Select the Model',['LogReg','DecisionTree','RandomForest'])

if model=="LogReg":
    model = lr
elif model == "DecisionTree":
    model = dt
else:
    model = rf



st.title('CFS vs Depression prediction')
st.write('Fill please the details for diagnostic Prediction')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider('Select Age',18,70,30)  # min=18,max=70, default = 30
    gender  = st.selectbox('Gender',['Male','Female'])
    sq_index = st.number_input('Sleep Quality Index',1.0,10.0,5.6)
    brain_fog = st.number_input('Brain Fog Level',1.0,10.0,6.4)
    pain_score = st.number_input('Physical Pain Score',1.0,10.0,4.5)

with col2:
    stress_level = st.number_input('Stress Level',1.0,10.0,4.8)
    dep_score = st.number_input('Depression phq9 Score',0,27,14)
    fat_sev_score = st.number_input('Fatigue Seveity Scale Score',0.3,10.0,3.5)
    pem_hrs = st.number_input('PEM Duartion Hrs',0,47,20)
    sleep_hrs = st.number_input('Sleep Hours',3.0,10.0,5.5)

with col3:
    pem_present = st.selectbox('PEM Present',[1,0])
    med_mind = st.selectbox('Meditation or Mindfulness',['Yes','No'])
    work_status = st.selectbox('Working_Status',['Working', 'Partially working', 'Not working'])
    social_act = st.selectbox('Social Activity Level',['Low', 'Very low', 'High', 'Medium', 'Very high'])
    exer_freq = st.selectbox('Exercise Frequency',['Daily', 'Often', 'Rarely', 'Never', 'Sometimes'])


gender = 1 if gender=="Male" else 0
med_mind = 1 if med_mind=="Yes" else 0

if work_status=="Working":
    ws_w = 1
    ws_pw = 0
    ws_nw = 0
elif work_status=="Partially working":
    ws_w = 0
    ws_pw = 1
    ws_nw = 0
else:
    ws_w = 0
    ws_pw = 0
    ws_nw = 1

if social_act == "Low":
    sa_l = 1
    sa_vl = 0
    sa_m = 0
    sa_h = 0
    sa_vh = 0
elif social_act == "Medium":
    sa_l = 0
    sa_vl = 0
    sa_m = 1
    sa_h = 0
    sa_vh = 0
elif social_act == "Very high":
    sa_l = 0
    sa_vl = 0
    sa_m = 0
    sa_h = 0
    sa_vh = 1
elif social_act == "Very low":
    sa_l = 0
    sa_vl = 1
    sa_m = 0
    sa_h = 0
    sa_vh = 0
else:
    sa_l = 0
    sa_vl = 0
    sa_m = 0
    sa_h = 1
    sa_vh = 0


if exer_freq == "Never":
    ef_n = 1
    ef_o = 0
    ef_r = 0
    ef_s = 0
    ef_d = 0
elif exer_freq == "Often":
    ef_n = 0
    ef_o = 1
    ef_r = 0
    ef_s = 0
    ef_d = 0
elif exer_freq == "Rarely":
    ef_n = 0
    ef_o = 0
    ef_r = 1
    ef_s = 0
    ef_d = 0
elif exer_freq == "Sometimes":
    ef_n = 0
    ef_o = 0
    ef_r = 0
    ef_s = 1
    ef_d = 0
else:
    ef_n = 0
    ef_o = 0
    ef_r = 0
    ef_s = 0
    ef_d = 1


test_data = [[age,gender,sq_index,brain_fog,pain_score,stress_level,dep_score,
              fat_sev_score,pem_hrs,sleep_hrs,pem_present,med_mind,
               ws_pw, ws_w,sa_l,sa_m,sa_vh,sa_vl, ef_n, ef_o, ef_r, ef_s]]


col_names = ['age', 'gender', 'sleep_quality_index', 'brain_fog_level',
       'physical_pain_score', 'stress_level', 'depression_phq9_score',
       'fatigue_severity_scale_score', 'pem_duration_hours',
       'hours_of_sleep_per_night', 'pem_present', 'meditation_or_mindfulness',
       'work_status_Partially working', 'work_status_Working',
       'social_activity_level_Low', 'social_activity_level_Medium',
       'social_activity_level_Very high', 'social_activity_level_Very low',
       'exercise_frequency_Never', 'exercise_frequency_Often',
       'exercise_frequency_Rarely', 'exercise_frequency_Sometimes']


test_df = pd.DataFrame(test_data,columns=col_names)

st.write('You have Entered the following')
st.write(test_df)


if st.button('Predict'):
    pred = model.predict(test_df)
    st.success(pred[0])