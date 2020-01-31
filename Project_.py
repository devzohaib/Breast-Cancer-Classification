# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:39:31 2020

@author: M Zohaib
"""


import time
import tkinter.messagebox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split    
from keras import Sequential
from keras.layers import Dense   
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix




# function to load data set
#def load_file():
#    #global file_path
#    file_path = filedialog.askopenfilename()
#    l2 = Label(mainframe,text=file_path)
#    l2.grid(column = 3,row=0)
#    print(file_path)
#    dataset = pd.read_csv(file_path)

# function of radio button implementation    
def sel():
    n = var1.get()
    l4 = Label(mainframe,text="You selected the option "+str(n))
    l4.grid(row=4,column = 0,sticky=W)
    if var1.get()==1:
       #Calling Naive Bayes fun
       Naive_bayes()
       print("navie bayes")
    else:
       #Calling Neural Network Function
       Neural_network()
       print("neural network")

       
counter =0
def counter_label(label):
    counter=0
    def count():
        global counter
        counter +=1
        print("counter"+str(counter))
        if counter==101:
            counter=0
            return 1
        label.config(text=str(counter)+" % ")
        label.after(5,count)
    stop=count() # when count return 1 it will stop
   

def train_navie():
    print("train_navie")          
    label = Label(mainframe,fg="dark green",text="counter")
    label.grid(row = 7, column = 0,sticky=W,padx=165)
    counter_label(label)

def test_navie():
    print("Test_navie")    
    labelt = Label(mainframe,fg="dark green",text="counter")
    labelt.grid(row = 9, column = 0,sticky=W,padx=165)
    counter_label(labelt)

def result_navie():
    
    filepath = 'cancer.csv'
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[:, 2:33].values
    Y = dataset.iloc[:, 1].values
    
    #print("Cancer data set dimensions : {}".format(dataset.shape))
    dataset.isnull().sum()
    dataset.isna().sum()
    
    #Encoding categorical data values

    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    
    # Splitting the dataset into the Training set and Test set

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    
    X_train=X_train[:,0:30] 
    X_test = X_test[:,0:30]
    
    #Fitting Naive_Bayes

    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    
    Y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(Y_test, Y_pred)
    #c = print(cm[0, 0] + cm[1, 1])
    #print(cm)
    #print(cm[0,0])
    #print(cm[0,1])
    #print(cm[1,0])
    #print(cm[1,1])
      
    total_prediction = str(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    
    """
    accuracy = no of correctly classifed samples/ total number of test samples
    Precision = no of true positive / total no of sample classified as positives
    
    """
    accuracy = ((TP+TN)/(TP+FP+FN+TN))*100
    precison = (TP / (TP+FN))*100
    
    accuracy = round(accuracy,3)
    precison = round(precison,3)
    
    t_p = str(TP)
    f_p = str(FP)
    f_n = str(FN)
    t_n = str(TN)
    
    accu = str(accuracy) 
    prec = str(precison)
    
    #print('Total Prediction          = '+total_prediction +'\n'+
    #      'True  Positive Prediction = '+t_p +'\n'+
    #      'True  Negative prediction = '+t_n +'\n'+
    #      'False Positive prediction = '+f_p+'\n'+
    #      'False Negative prediction = '+f_n+'\n'+
    #      ' \n'+
    #      'Accuracy  = ' +accu + ' %\n'+
    #      'Precision = ' +prec + ' %'
    #      )
    
    #print("Totoal Prediction = "+total_prediction)
    #print("True Positvie prediction = "+true_positive)
    
    tkinter.messagebox.showinfo('Results','Total Prediction                  = '+total_prediction +'\n'+
      'True  Positive Prediction    = '+t_p +'\n'+
      'True  Negative prediction  = '+t_n +'\n'+
      'False Positive prediction    = '+f_p+'\n'+
      'False Negative prediction  = '+f_n+'\n'+
      ' \n'+
      'Accuracy  = ' +accu + ' %\n'+
      'Precision = ' +prec + ' %'
      )
    
       
def Naive_bayes():
       space_l = Label(mainframe, text ="   ",font =('Arial',8,))
       space_l.grid(row=6,column = 0,sticky=W)
       
       train_model = Button(mainframe, text = "Train_Navie_model",font =('Arial',12),bd=3,width=16,command = train_navie)
       train_model.grid(row = 7, column = 0,sticky=W)
       
       space_l = Label(mainframe, text ="   ",font =('Arial',8,))
       space_l.grid(row=8,column = 0,sticky=W)
       
       train_model = Button(mainframe, text = "Test_Navie_model",font =('Arial',12),bd=3,width=16,command = test_navie)
       train_model.grid(row = 9, column = 0,sticky=W)
       
       space_l2 = Label(mainframe, text ="   ",font =('Arial',8,))
       space_l2.grid(row=10,column = 0,sticky=W)
       
       show_result = Button(mainframe,text="Result",font =('Arial',12),bd=3,width=16,command= result_navie)
       show_result.grid(row=11,column =0, sticky=W)

def train_neural():
    print("train_neural")          
    label = Label(mainframe,fg="dark green",text="counter")
    label.grid(row = 7, column = 0,sticky=W,padx=165)
    counter_label(label)
    
def test_neural():
    print("Test_neural")    
    labelt = Label(mainframe,fg="dark green",text="counter")
    labelt.grid(row = 9, column = 0,sticky=W,padx=165)
    counter_label(labelt)

def result_neural():

    filepath = 'cancer.csv'
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[:, 2:32].values # 30 input features
    Y = dataset.iloc[:, 1].values
    
    dataset.isnull().sum()
    dataset.isna().sum()
    
    #standardizing the input feature
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    #Encoding categorical data values
    
    
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal', input_dim=30))
    #Second  Hidden Layer
    classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    
    #Compiling the neural network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    
    #Fitting the data to the training dataset
    classifier.fit(X_train,y_train, batch_size=10, epochs=50)
    
    eval_model=classifier.evaluate(X_train, y_train)
    y_pred=classifier.predict(X_test)
    y_pred =(y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
   # print(cm)
    total_prediction = str(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    
    accuracy = ((TP+TN)/(TP+FP+FN+TN))*100
    precison = (TP / (TP+FN))*100
    
    accuracy = round(accuracy,3)
    precison = round(precison,3)
    
    t_p = str(TP)
    f_p = str(FP)
    f_n = str(FN)
    t_n = str(TN)
    
    accu = str(accuracy) 
    prec = str(precison)
    
    tkinter.messagebox.showinfo('Results','Total Prediction                  = '+total_prediction +'\n'+
      'True  Positive Prediction    = '+t_p +'\n'+
      'True  Negative prediction  = '+t_n +'\n'+
      'False Positive prediction    = '+f_p+'\n'+
      'False Negative prediction  = '+f_n+'\n'+
      ' \n'+
      'Accuracy  = ' +accu + ' %\n'+
      'Precision = ' +prec + ' %'
      )

    
    
def Neural_network():    
       space_l = Label(mainframe, text ="   ",font =('Arial',8,))
       space_l.grid(row=6,column = 0,sticky=W)
       
       train_model = Button(mainframe, text = "Train_Neural_model",font =('Arial',12),bd=3,width=16,command = train_neural)
       train_model.grid(row = 7, column = 0,sticky=W)
       
       space_l = Label(mainframe, text ="   ",font =('Arial',8,))
       space_l.grid(row=8,column = 0,sticky=W)
       
       train_model = Button(mainframe, text = "Test_Neural_model",font =('Arial',12),bd=3,width=16,command = test_neural)
       train_model.grid(row = 9, column = 0,sticky=W)
       
       space_l2 = Label(mainframe, text ="   ",font =('Arial',8,))
       space_l2.grid(row=10,column = 0,sticky=W)
       
       show_result = Button(mainframe,text="Result",font =('Arial',12),bd=3,width=16,command = result_neural)
       show_result.grid(row=11,column =0, sticky=W)
     
"""
main Window coding 
"""        
root = Tk()
root.title("Application")
mainframe = LabelFrame(root, text="Classification of Breast Cancer",bd =3,font =('Arial',25,"italic"),padx=5, pady=5)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

#l1 = Label(mainframe, text = "Browse to Load the data Set.",font =('Arial',12,))
#l1.grid(row = 0, column = 0,pady = 20,sticky=W)

#b1 = Button(mainframe, text = "Browse",font =('Arial',12),bd=3,command = load_file)
#b1.grid(row = 0, column = 2)

#                                       #
# label for selecting Analysis Approach #
#_____________________________________  #
l3 = Label(mainframe, text ="Select Your Analysis Approach!",font =('Arial',18,))
l3.grid(row=0,column = 0,sticky=W,pady=20)
lb = Label(mainframe, text ="   ",font =('Arial',8,))
lb.grid(row=1,column = 0,sticky=W)
#                                             #
# Radio Button button for selection of model  #
##_____________________________________       # 
var1 = IntVar()
R1 = Radiobutton(mainframe, text="Option 1:Naive Bayes",font =('Arial',10,) ,   variable=var1, value=1,command=sel)
R1.grid(row=2,column = 0,sticky=W)

R2 = Radiobutton(mainframe, text="Option 2:Neural Network",font =('Arial',10,), variable=var1, value=2,command=sel)
R2.grid(row=3,column = 0,sticky=W)    


root.mainloop()