import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D

# read the csv file
def read_sample_list(folder_name):
    samples = []
    with open(folder_name + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
            
    # split the training and validation data
    train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
    
    print('Detected {:d} samples'.format(len(train_samples)))
    return train_samples, validation_samples

def generator(samples, folder_name, steering_correction=2, augment=False, batch_size=32):
    num_samples = len(samples)
    
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                name = folder_name + '/IMG/' + batch_sample[0].split('\\')[-1]
                #name = batch_sample[0]
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                if (augment):
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)
                    namel = folder_name + '/IMG/' + batch_sample[1].split('\\')[-1]
                    namer = folder_name + '/IMG/' + batch_sample[2].split('\\')[-1]
                    #namel = batch_sample[1]
                    #namer = batch_sample[2]
                    left_image = cv2.cvtColor(cv2.imread(namel), cv2.COLOR_BGR2RGB)
                    right_image = cv2.cvtColor(cv2.imread(namer), cv2.COLOR_BGR2RGB)
                    images.append(left_image)
                    angles.append(center_angle + steering_correction)
                    images.append(right_image)
                    angles.append(center_angle - steering_correction)
                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        
def get_model(load_file_name=None):
    if (load_file_name is not None):
        model = load_model(load_file_name)
    else:
        ch, row, col = 3, 160, 320
        model = Sequential()
        model.add(Cropping2D(cropping=((50,10), (0,0)), input_shape=(row, col, ch)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
        model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
        model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
        model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
    return model

def run_model(model, sample_folder_name, save_file_name=None, steering_correction=2, nb_epoch=3):
    train_samples, validation_samples = read_sample_list(sample_folder_name)    
    
    train_generator = generator(train_samples, sample_folder_name, steering_correction, augment=True, batch_size=32)
    validation_generator = generator(validation_samples, sample_folder_name, steering_correction, augment=False, batch_size=32)
    
    history_object = model.fit_generator(train_generator,\
                                         samples_per_epoch=len(train_samples)*4,\
                                         validation_data=validation_generator,\
                                         nb_val_samples=len(validation_samples),\
                                         nb_epoch=nb_epoch)
    if (save_file_name is not None):
        model.save(save_file_name)

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    
    return model

model = get_model(None)
model = run_model(model, './01_03_laps', '01_03_model_3eps.h5', nb_epoch=3)

#float(model.predict(image_array[None, :, :, :], batch_size=1))