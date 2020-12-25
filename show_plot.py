
# %%
import time
import pickle
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results_file = ['out_corner','open','one_wall']##['corridor','l_corridor','plus_corridor']#,

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
# %%
for i in range(0,len(all_data)):  
    full_data = []
    full_data = all_data[i]
    tot1 = 0
    x=1
    for a in full_data[0]:  # a = 90
        if (i==2 and x >20 and x <30 ):
            a = np.array(a) 
            a = a[:len(a)-1] # remove the label
            plt.figure(figsize=(20,10))
            plt.polar([(float(i)/180)*np.pi for i in range(0,360)], a,'*')
        x+=1
    
# %%
