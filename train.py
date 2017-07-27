import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

# read the csv file
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]
        
# split the training and validation data
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/' + batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        
       
"""
images = []
measurements = []
# consider using left/rigth images
for line in lines[1:]:
    source_path = './data/' + line[0]
    image = cv2.imread(source_path)
    images.append(image)
    images.append(np.fliplr(image))
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(-measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape, y_train.shape)"""


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, \
                                     samples_per_epoch=len(train_samples), \
                                     validation_data=validation_generator, \
                                     nb_val_samples=len(validation_samples), \
                                     nb_epoch=3)
model.save('model.h5')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()