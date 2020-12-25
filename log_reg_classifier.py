
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def collate_data():
    results_file = ['one_wall_fv','open_fv','out_corner_fv','corridor_fv','l_corridor_fv','plus_corridor_fv']

    train_data = np.empty((0,7),float)
    test_data = np.empty((0,7),float)
    train_label = np.empty((0,),float)
    test_label = np.empty((0,),float)

    for r in results_file:
        data = []
        with (open(r,"rb")) as fp:
            while True:
                try:
                    data.append(pickle.load(fp))
                except EOFError:
                    break
        data = data[0]
        
        Xdata = []
        Ydata = []
        for a in data:  # a = 89
            a = np.array(a) 
            Ydata = np.append(Ydata, a[len(a)-1])
            a = a[:len(a)-1] # remove the label
            Xdata.append(list(a))
        
        X_train,X_test,Y_train,Y_test = train_test_split(Xdata,Ydata,test_size=0.20,random_state=0)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        train_data = np.append(train_data,X_train,axis=0)
        test_data = np.append(test_data,X_test,axis=0)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        train_label = np.append(train_label,Y_train,axis=0)
        test_label = np.append(test_label,Y_test,axis=0)

    #print(np.shape(train_data))
    #print(np.shape(test_data))
    return train_data,test_data,train_label,test_label

def classify(train_data,test_data,train_label,test_label):
    logreg = LogisticRegression()
    logreg.fit(train_data,train_label)
    y_pred=logreg.predict(test_data)
    cnf_matrix = metrics.confusion_matrix(test_label, y_pred)
    print(cnf_matrix)

if __name__ == '__main__':

    train_data,test_data,train_label,test_label = collate_data()
    classify(train_data,test_data,train_label,test_label)