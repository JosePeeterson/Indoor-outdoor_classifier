
#  baseline classification accuracy

import time
import pickle
import numpy as np

def collate_data():
    results_file = ['one_wall']#['one_wall','corridor','l_corridor','plus_corridor','out_corner','open','one_wall']

    all_data = []
    for r in results_file:
        data = []
        with (open(r,"rb")) as fp:
            while True:
                try:
                    data.append(pickle.load(fp))
                except EOFError:
                    break
        all_data.append(data)
    
    print(len(all_data[0][0]),len(all_data[0][0][0]))
    print(all_data)
    return all_data


def baseline_classifier(all_data):# 6*90*361
    for i in range(0,len(all_data)):  
        full_data = []
        full_data = all_data[i]
        tot1 = 0
        for a in full_data[0]:  # a = 90
            a = np.array(a) 
            a = a[:len(a)-1] # remove the label
            tot1 = tot1 + len(a[a<5])

        #print(float(tot1)/(len(full_data[0])*360))


if __name__ == '__main__':

    all_data = collate_data()
    baseline_classifier(all_data)