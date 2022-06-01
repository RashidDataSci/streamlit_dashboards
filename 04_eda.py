# import libraries
import csv
from distutils.command.upload import upload
from doctest import Example
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# web app ka title
st.markdown('''
# **Exploratory Data Analysis web application**
This app is developed by codanics youtube channel called **EDA App**
    ''')

# how to upload a file from pc

with st.sidebar.header(' Upload your dataset (.csv)'):
    upload_file = st.sidebar.file_uploader('Upload your file', type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown('[Example CSV file](https://raw.githubusercontent.com/divmain/GitSavvy/master/README.md)')

# profiling report for pandas

if upload_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(upload_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    st.header('**Profiling report with pandas**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file, upload kar b do ab ya kaam nahi lena?')
    if st.button('Press to use example data'):
        # example data set
        @st.cache
        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),
                                columns=['age','banana','codanics','Deutchland','Ear'])
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DF**')
        st.write(df)
        st.write('---')
        st.header('**Profiling report with pandas**')
        st_profile_report(pr)
