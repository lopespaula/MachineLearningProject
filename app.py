import streamlit as st
import plotly_express as px
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Clothing classification App")

st.sidebar.subheader("File uploader")

uploaded_file = st.sidebar.file_uploader(label="Upload your .PNG file.",
                    type = ['png'])
                    
if uploaded_file is not None:
    print(uploaded_file)
    print("hello")
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)

