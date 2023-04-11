import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def stats(dataframe):
    st.header('Data Statistics')
    st.write(dataframe.describe())

def data_header(dataframe):
    st.header('Data Header')
    st.write(df.head())

def plot(dataframe):
    fig, ax = plt.subplot(1,1)
    st.pyplot(fig)

st.title('Platform_Pinlifter_B2_Accu-Sync')
st.text('This is a web app to explore the accuracy and synchronisity measurement of the PP_B2')

st.sidebar.title('Navigation')
uploaded_file = st.sidebar.file_uploader('Upload your file here')

options = st.sidebar.radio('Pages', options=['Home', 'Data Statistics', 'Data Header', 'Plot'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
if options == 'Data Statistics':
    stats(df)
elif options == 'Data Header':
    data_header(df)
elif options == 'Plot':
    plot(df)