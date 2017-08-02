import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras import regularizers
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

# generate batch samples from the list of samples	
def generator(samples, folder_name, steering_correction=2, augment=False, batch_size=32):
    num_samples = len(samples)
    
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                # center image to the batch sample
                name = folder_name + '/IMG/' + batch_sample[0].split('\\')[-1]
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
				  # If augment is enabled, add a few extra augmented data
				  # 1. mirror image of the center image with negative steering angles
				  # 2. left camera image with steering correction added
				  # 3. right camera image with steering correction added
				  # Note: batch size quadruples if augment==True
                if (augment):
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)
                    namel = folder_name + '/IMG/' + batch_sample[1].split('\\')[-1]
                    namer = folder_name + '/IMG/' + batch_sample[2].split('\\')[-1]
                    left_image = cv2.cvtColor(cv2.imread(namel), cv2.COLOR_BGR2RGB)
                    right_image = cv2.cvtColor(cv2.imread(namer), cv2.COLOR_BGR2RGB)
                    images.append(left_image)
                    angles.append(center_angle + steering_correction)
                    images.append(right_image)
                    angles.append(center_angle - steering_correction)
                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# open or generate the neural network			
def get_model(load_file_name=None):

    # load a saved model if the file name is provided
    if (load_file_name is not None):
        model = load_model(load_file_name)
    else:
		 # generate the model
        ch, row, col = 3, 160, 320
        model = Sequential()
		
		 # crop the top and bottom of the image (sky, car hood)
        model.add(Cropping2D(cropping=((50,10), (0,0)), input_shape=(row, col, ch)))
		
		 # pre-process / normalize the image
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
		
		 # Add 5 convolutional layers with relu activation
        model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu', W_regularizer=regularizers.l2(0.0001)))
        model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
        model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(Conv2D(64, 3, 3, activation='relu'))

		 # Add 3 fully connected layers with relu activation
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))

        # Add the output layer
        model.add(Dense(1))
		
        # Use MSE as loss function and use Adam optimizer
        model.compile(loss='mse', optimizer='adam')

    return model

# run the model with the given parameters
def run_model(model, sample_folder_name, save_file_name=None, steering_correction=2, nb_epoch=3, batch_size=32):
    # get the list of samples from the given folder
    train_samples, validation_samples = read_sample_list(sample_folder_name)    
    
	 # get batch samples from the generator
    train_generator = generator(train_samples, sample_folder_name, steering_correction, augment=True, batch_size=batch_size)
    validation_generator = generator(validation_samples, sample_folder_name, steering_correction, augment=False, batch_size=batch_size)
    
    # fit the model.
    # samples per epoch is quadrupled due to augmentation
    history_object = model.fit_generator(train_generator,\
                                         samples_per_epoch=len(train_samples)*4, \
                                         validation_data=validation_generator,\
                                         nb_val_samples=len(validation_samples),\
                                         nb_epoch=nb_epoch)
    
    # save the results
    if (save_file_name is not None):
        model.save(save_file_name)

	 # graph history
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    
    return model

model = get_model(None)
model = run_model(model, './01_slow_lap', 'model.h5', steering_correction=5, nb_epoch=5, batch_size=32)
