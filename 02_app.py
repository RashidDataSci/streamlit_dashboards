from multiprocessing.sharedctypes import Value
from click import option
from requests import options
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# make container
header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Kashti ki app')
    st.text('In the project we will work on kashti data.')
with data_sets:
    st.header('Kashti doob gaye. Hawww!')
    st.text('we will work with Titanic datasets')
    # import data set
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))

    st.subheader('Sambha, Are oooh sambha, kitnay aadmi thay')
    st.bar_chart(df['sex'].value_counts())

    # other plots
    st.subheader('class k hasab sy farak')
    st.bar_chart(df['class'].value_counts())

    # barplot
    st.bar_chart(df['age'].sample(10)) # or .head(10)


with features:
    st.header('These are our app features')
    st.text('Awen bht saray features add kartay hyn, asaaan hi hy')
    st.markdown('1. **Featuress 1:** This will tell us pata nahi')
    st.markdown('2. **Featuress 1:** This will tell us pata nahi')


with model_training:
    st.header('kashti walo ka kia bana?-Model training')
    st.text('Is main ham apny paper ko kam ya zayada kr sakty hy')

# making columns
input, display = st.columns(2)

# pehlay column main ap k selection points hun
max_depth = input.slider('How many people do you know?', min_value=10, max_value=100, value=20, step=5)
#n_estimator
n_estimators = input.selectbox('How many tree should be there in a RF?', options=[50,100,200,300,'NO limit'])

# adding list of features
input.write(df.columns)

# input features from user
input_features = input.text_input('Which feature we should use?')

# machine learning model
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# yahan per ham aik condition lagayen gay
if n_estimators =='NO limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# define x and y
x = df[[input_features]]
y = df[['fare']]

# fit our model
model.fit(x, y)
pred = model.predict(y)

# display matrices
display.subheader('Mean absolute error of the model is: ')
display.write(mean_absolute_error(y, pred))
display.subheader('Mean squared error of the model is: ')
display.write(mean_squared_error(y, pred))
display.subheader('R squared score of the model is: ')
display.write(r2_score(y, pred))