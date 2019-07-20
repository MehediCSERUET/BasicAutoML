from flask import Flask, render_template, request
from werkzeug import secure_filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
app = Flask(__name__)

def calculate_performance(y_pred, y_test):
   TP = FP = TN = FN = 0
   for i in range(len(y_test)):
      if int(y_pred[i]) == y_test[i] and y_test[i] == 1:
         TP = TP + 1
      elif int(y_pred[i]) == y_test[i] and y_test[i] == 0:
         TN = TN + 1
      elif int(y_pred[i]) is not y_test[i] and y_test[i] == 0:
         FP = FP + 1
      elif int(y_pred[i]) is not y_test[i] and y_test[i] == 1:
         FN = FN + 1
   return (TP, TN, FP, FN)

@app.route('/process', methods = ['POST', 'GET'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))

      ratio = int(request.form.get('testRatio'))/100.0
      algo = request.form.get('algoSelector')
      #colNamePresent = request.form.get('checker')

      if algo == "LR":
         model = LinearRegression()
      elif algo == "NB":
         model = GaussianNB()
      elif algo == "KNN":
         model = KNeighborsClassifier(n_neighbors=3)
      elif algo == "LDA":
         model = LinearDiscriminantAnalysis()
      elif algo == "LoR":
         model = LogisticRegression()
      elif algo == "SVM":
         model = SVC()
      
      dataframe = pd.read_csv(f.filename)
      array = dataframe.values
      N_Col = len(array[0])
      N_Col = N_Col - 1 #since array starts at 0
      X = array[:, 0:N_Col-1] #assuimng last column has target value
      Y = array[:, N_Col]
      
      X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=0)
      history = model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      #score = model.evaluate(X_test, y_test)
      (TP, TN, FP, FN) = calculate_performance(y_pred, y_test)
      try:
         accuracy = (TP+TN)/(TP+TN+FP+FN)
         recall = TP/(TP+FN)
         precision = TP/(TP+FP)
         f1_score = 2*((precision * recall)/(precision + recall))
      except:
         return "<h3>TP: "+str(TP) + "</h3><br><h3>TN: "+str(TN)+"</h3><br><h3>FP: "+str(FP)+"</h3><br><h3>FN: "+str(FN)+"</h3>"
      return  "<h3>Accuracy: " + str(accuracy * 100) + "</h3><br><h3>Precision: " + str(precision * 100) + "</h3><br><h3>Recall: " + str(recall * 100) + "</h3><br><h3>F1 Score: " + str(f1_score * 100) + "</h3>"

if __name__ == '__main__':
    app.run(debug=True)