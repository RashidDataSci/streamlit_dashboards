# import libraries
from datetime import date
import imp
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app ki heading
st.write('''
# Explore different ML Models and Data sets
Daikhty hain kon sa best hain in main sy?
''')

# dat set k nam aik box main dal kr side pr laga do
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# or isi k nichy classifier k nam daby main dal do
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN','SVM', 'Random Forest')
)

# ab hum nay aik function define krna hy data set ko load krny k liay
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name =='Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y 

# ab is function ko bula lay gy or x, y variable k equal rakh ly gy
x, y = get_dataset(dataset_name)

# ab hum apny data set ki shape ko ap pay print kr dan gy 
st.write('Shape of dataset:', x.shape) 
st.write('Number of classes:', len(np.unique(y)))

# next hum different classifier k parameter k user input main add kray gy
def add_parameter_ui(classifier_name):
    params = dict() # create an empty dictionary
    if classifier_name == 'SVM':
        c = st.sidebar.slider('c', 0.01, 10.0)
        params['c'] = c # its the degree of correct classification
    elif classifier_name == 'KNN':
        k = st.sidebar.slider('k',1,15)
        params['k'] = k # its the number of nearest neighber
    else:
        max_depth = st.sidebar.slider('max_depth',2, 15)
        params['max_depth'] = max_depth # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators # number of tree
    return params

# ab is function ko bula ly gy or params variable k equal rakh ly gy
params = add_parameter_ui(classifier_name)

# ab hum classifier bnay gy base on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['c'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], random_state=1234)
    return clf

# ab is function ko bula ly gy or clf variable k equal k rakh ly gy
clf = get_classifier(classifier_name, params)

# ab hum apny dataset ko train test main split kr ly gy by 80/20 ratio
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1234)

# ab hum ny apny classifier ki training krni hy
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# model ki accuracy check kr lain gy or is ko app pr print kr lay gy
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', accuracy)

#### Plot Dataset ####
# ab hum apnay saray saray features ko 2 dimensional plot pay draw kr dy gy using pca
pca = PCA(2)
x_projected = pca.fit_transform(x)

# ab hum apny data 0 or 1 dimension ma slice kr dy gy
x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)
