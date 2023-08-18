import streamlit as st 
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Streamlit Example")

st.write("""
# Explore Different Classifier

""")

dataset_name = st.sidebar.selectbox("Select Dataset" , ("Iris" , "Breast Cancer" , "Wine dataset"))
# st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier" , ("KNN" , "SVM" , "Random Forest"))
st.write(classifier_name)

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else : 
        data = datasets.load_wine()
    
    x = data.data
    y = data.target
    return x , y

x , y = get_dataset(dataset_name)

st.write("Dataset Name :" , dataset_name)
st.write("shape of x :" , x.shape , 'shape of y :',y.shape)
st.write("number of classes" , len(np.unique(y)))

def add_parameters(classifier):
    param = dict()
    if classifier == "KNN":
        k = st.sidebar.slider("K" , 1 , 15)
        param["k"] = k
    elif classifier == "SVM":
        c = st.sidebar.slider("c" , 0.01 , 10.0)
        param["c"] = c
    else : 
        max_depth = st.sidebar.slider("Maximum depth" , 2 , 15)
        n_est = st.sidebar.slider("Number of Estimators" , 1 , 100)
        param["max_depth"] = max_depth
        param["n_est"] = n_est 
    
    return param
    
parameters = add_parameters(classifier_name)


def get_classifier(classifier , param):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=param["k"])
    elif classifier == "SVM":
        clf = SVC(C = param["c"])
    else : 
        clf = RandomForestClassifier(n_estimators=param["n_est"] , max_depth=param["max_depth"] , random_state=1234)

    return clf 

clf = get_classifier(classifier_name , parameters)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=1234)

clf.fit(x_train , y_train)

y_predict = clf.predict(x_test)

acc = accuracy_score(y_test , y_predict)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy : {acc}')

##PLOT 
pca = PCA(2)
x_proj = pca.fit_transform(x)

x1 = x_proj[:,0]
x2 = x_proj[:,1]

fig = plt.figure()
plt.scatter(x1 , x2 , c = y , alpha = 0.8 , cmap = 'viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)