import streamlit as st 

st.title("Machine Learning Based Analysis of Lyme Disease")

st.markdown("This projects presents two machine learning based approaches to analyzing lyme disease. " + 
        "The project has been split into to main parts: **lyme disease case rate prediction** and **lyme disease identification**. "
        + "Lyme disease case rate prediction was implemented using time series analysis, while lyme disease identification was implemented using CNN analysis on images of skin lesions. ")


st.subheader("CNN Analysis of Skin Lesions")
st.markdown('This part of the project presents a CNN based classifier of skin lesions. Lyme disease often results in very distintive bullseye skin lesions on the body. We implemented a CNN to predict whether an image of a skin lesion is a case of Lyme disease or not. The CNN was manually implemented using R and uses an edge detection kernel to determine whether a skin lesion looks like Lyme disease')


st.subheader("Time Series Analysis of Lyme Disease Case Rates")
st.markdown("This part of the project presents a time series model based analysis of Lyme disease cases. AR, MA, ARMA, and SARIMA models were used to analyze weekly Lyme disease case rate data to predict future rates of Lyme disease. The AR model achieved the best performance of the four time series models with a MAE of ~ 130. ")