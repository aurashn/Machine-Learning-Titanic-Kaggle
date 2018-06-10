import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential,save_model
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.utils import normalize 


num_classes = 2

def prep_data(raw, train_size, val_size):
    y = raw[:, 1]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x1 = raw[:,2]
    x2 = raw[:,5]
    x3 = raw[:,6]
    x4 = raw[:,7]
    x5 = raw[:,8]
    x6 = raw[:,10]
    x7 = raw[:,12]
    sum1 = 0
    count = 0 
    for val in x4:
    	if val != '':
    		sum1+=float(val)
    		count+=1	
    sum1/=count
    countx2 = 0 
    for val in x2:
    	if val == "male":
    		x2[countx2] = 0
    	else:
    		x2[countx2] = 1
    	countx2+=1
    countx7 = 0 
    for val in x7:
    	if val == "S":
    		x7[countx7] = 0
    	elif val == "C":
    		x7[countx7] = 1
    	elif val == "Q":
    		x7[countx7] = 2	
    	countx7+=1
    out_x = np.column_stack((x1,x2,x3,x4,x5,x6,x7))
    for j in range(np.size(out_x,0)):
    	for k in range(np.size(out_x,1)):
    		print out_x[j][k]
    		if out_x[j][k] == '':
    			out_x[j][k] = sum1
    		out_x[j][k] = float(out_x[j][k])
    out_x = preprocessing.scale(out_x)		 
    out_x = out_x.reshape(891,1,7,1)
    return out_x, out_y

titanic_file = "train.csv"
titanic_data = np.loadtxt(titanic_file, dtype ='str',skiprows=1, delimiter=',')
x, y = prep_data(titanic_data, train_size=50000, val_size=5000)
print(x)
titanic_model = Sequential()
titanic_model.add(Dense(50,activation='relu', input_shape = (1,7,1)))
titanic_model.add(Dense(200,activation='relu'))
titanic_model.add(Dense(200,activation='relu'))
titanic_model.add(Dense(50,activation='relu'))
titanic_model.add(Dense(50,activation='relu'))
titanic_model.add(Flatten())
titanic_model.add(Dense(num_classes,activation = 'sigmoid'))

titanic_model.compile(loss = keras.losses.binary_crossentropy,optimizer = 'adam', metrics = ['accuracy'])

titanic_model.fit(x,y, batch_size = 500, epochs = 100, validation_split = 0.1)
save_model(titanic_model,"/home/aurash/titanic_kaggle/model.hdf5",overwrite = True, include_optimizer=True)
