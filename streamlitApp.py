import streamlit as st
import pandas as pd
import os
import joblib
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json

def showModelEval(modelDataPathR):
    
    #show scoring
    modelDataPath = modelDataPathR + ".json"
    with open(modelDataPath) as file:
        modelData = json.load(file)
    
    st.write("mean of 10 accuracies, gotten durning k-cross validation")
    mean10Acc = str(round(modelData["mean10Acc"], 4))
    mean10Acc = '<p style="font-size: 32px; color: Green">{}</p>'.format(mean10Acc)
    st.markdown(mean10Acc, unsafe_allow_html=True)
    
    st.write("mean of 10 recalls, gotten durning k-cross validation")
    mean10Rec = str(round(modelData["mean10Rec"], 4))
    mean10Rec = '<p style="font-size: 32px; color: Green">{}</p>'.format(mean10Rec)
    st.markdown(mean10Rec, unsafe_allow_html=True)
    
    st.write("standard deviation of 10 accuracies, gotten durning k-cross validation")
    accStd = str(round(modelData["accStd"], 4))
    accStd = '<p style="font-size: 32px; color: Green">{}</p>'.format(accStd)
    st.markdown(accStd, unsafe_allow_html=True)
    
    st.write("standard deviation of 10 recalls, gotten durning k-cross validation")
    recStd = str(round(modelData["recStd"], 4))
    recStd = '<p style="font-size: 32px; color: Green">{}</p>'.format(recStd)
    st.markdown(recStd, unsafe_allow_html=True)
    
    st.write("accuracy gotten durning final test on the test set")
    acc = str(round(modelData["acc"], 4))
    acc = '<p style="font-size: 32px; color: Green">{}</p>'.format(acc)
    st.markdown(acc, unsafe_allow_html=True)
    
    st.write("recall gotten durning final test on the test set")
    rec = str(round(modelData["rec"], 4))
    rec = '<p style="font-size: 32px; color: Green">{}</p>'.format(rec)
    st.markdown(rec, unsafe_allow_html=True)
    
    #show confusion matrix
    modelCmPath = modelDataPathR + "Cm.json"
    with open(modelCmPath) as file:
        modelCm = json.load(file)
    
    modelCm = [[modelCm["tn"], modelCm["fp"]],
               [modelCm["fn"], modelCm["tp"]]]

    modelCm = np.array(modelCm)
    
    fig, ax = plt.subplots(figsize = (3, 3))
    ax.matshow(modelCm)
    
    for (i, j), value in np.ndenumerate(modelCm):
        ax.text(j, i, "{}".format(value), ha='center', va='center')

    buf = BytesIO()
    fig.savefig(buf, format = "png")
    st.image(buf)

def getUserInput():
    userData = []
    labels = ["clump thickness", "uniformity of cell size",
              "uniformity of cell shape", "marginal adhesion",
              "single epithelial cell size", "bare nuclei",
              "bland chromatin", "normal nucleoli",
              "mitoses"]
    for label in labels:
        userData.append(int(st.text_input(label, 0)))
    return np.array(userData).reshape(1, 9)

def getModelPrediction(modelPath, userData):
    model = joblib.load(modelPath)
    return model.predict(userData)


#conventional way of getting the file path doesn't work with streamlit so this is used
projectDirPath = os.path.dirname(os.path.abspath(__file__))

#General info
st.header("Breast cancer tumor classification")
st.write("In this app you can view the analysis of breast cancer tumor" + 
         " classification problem using three machine learning models:")
st.markdown("__- Logistic Regression__")
st.markdown("__- Support Vector Machine__")
st.markdown("__- K Nearest Neighbors__")
st.write("it is also possible to enter your own values of features, and" +
         " get the prediction using all three models")

#Dataset overview
st.header("Original dataset overview")

datasetPath = projectDirPath + "/original data and data description/breast-cancer-wisconsin.csv"
dataset = pd.read_csv(datasetPath)

newNames = {"1000025" : "Sample code number",
            "5" : "Clump Thickness",
            "1" : "Uniformity of Cell Size",
            "1.1" : "Uniformity of Cell Shape",
            "1.2" : "Marginal Adhesion",
            "2" : "Single Epithelial Cell Size",
            "1.3" : "Bare Nuclei",
            "3" : "Bland Chromatin",
            "1.4" : "Normal Nucleoli",
            "1.5" : "Mitoses",
            "2.1" : "class"}
dataset = dataset.rename(columns = newNames, inplace = False)
st.write(dataset)
st.write("sample code number - irrelevant for classification (removed in preprocessed data)")
st.write("Class: 4 - malignant, 2 - benign")

#Logistic Regression analysis
st.header("Logistic Regression Analysis")
st.write("Two models were trained, the one with better recall was chosen"
        + " because for this specific task recall is more important than accuracy")

col1Log, col2Log = st.columns(2)
with col1Log:
    st.subheader("Logistic Regression - Linear border")
    showModelEval(projectDirPath + "/modelsData/logisticRegression")
with col2Log:
    st.subheader("Logistic Regression - Non-linear border")
    showModelEval(projectDirPath + "/modelsData/logisticRegressionPoly")
st.markdown('<p style="font-size: 28px; color: Red">linear border ' +
            'was chosen for its better recall</p>', unsafe_allow_html=True)

#svc analysis
st.header("Support Vector Classifier")
st.write("One model with linear kernel was trained")
showModelEval(projectDirPath + "/modelsData/svc")

#knn analysis
st.header("K Nearest Neighbors Analysis")
st.write("Three models were trained, with 3, 4 and 5 neighbors. " + 
         "The one with the highest recall was chosen")

col1Knn, col2Knn, col3Knn = st.columns(3)

with col1Knn:
    st.subheader("K-NN, 3 neighbors")
    showModelEval(projectDirPath + "/modelsData/knn1")
with col2Knn:
    st.subheader("K-NN, 4 neighbors")
    showModelEval(projectDirPath + "/modelsData/knn2")
with col3Knn:
    st.subheader("K-NN, 5 neighbors")
    showModelEval(projectDirPath + "/modelsData/knn3")
st.markdown('<p style="font-size: 28px; color: Red">5 neighbors ' +
            'were chosen for its better recall</p>', unsafe_allow_html=True)

#take user input and get prediction for every model that was chosen
st.header("prediction based on user input")

with st.form("user input"):
    userInput = getUserInput()
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.subheader("Logistic regression prediction")
        logPath = projectDirPath + "/models/logisticRegression.joblib"
        logPred = getModelPrediction(logPath, userInput)
        if (logPred == 4):
            st.write("the tumor is Malignant")
        else:
            st.write("the tumor is benign")
        
        st.subheader("Support vector classifier")
        svcPath = projectDirPath + "/models/svc.joblib"
        svcPred = getModelPrediction(svcPath, userInput)
        if (svcPred == 4):
            st.write("the tumor is Malignant")
        else:
            st.write("the tumor is benign")
        
        st.subheader("K nearest neighbors")
        knnPath = projectDirPath + "/models/K-NN.joblib"
        knnPred = getModelPrediction(knnPath, userInput)
        if (svcPred == 4):
            st.write("the tumor is Malignant")
        else:
            st.write("the tumor is benign")