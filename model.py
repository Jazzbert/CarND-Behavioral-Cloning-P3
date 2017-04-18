import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# global variables
DATADIR = './data/'
IMGDIR = DATADIR + 'IMG/'
CORRECTION_FACTOR = 0.2

# read csv file into lines
lines = []
with open(DATADIR + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

# generator to extract the images and steering angles
def generator(samples, batch_size=32):
    num_samples = len(samples)
    adjustment = [0.0, CORRECTION_FACTOR, -CORRECTION_FACTOR]

    while 1:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for sample in batch_samples:
                for i in range(3):
                    source_path = sample[i]
                    filename = source_path.split('/')[-1]
                    img_path = IMGDIR + filename
                    # add normal image and angle
                    image = cv2.imread(img_path)
                    images.append(image)
                    angle = float(sample[3]) + adjustment[i]
                    angles.append(angle)
                    # add flipped image and angle
                    images.append(cv2.flip(image,1))
                    angles.append(angle * -1.0)

            X = np.array(images)
            #X = X[:,70:135,:,:]
            y = np.array(angles)
            yield X, y

print("Total training samples: {}".format(len(train_samples)))

# start keras model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout

# use generator to train model
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
row, col, ch = 160, 320, 3

# initiallize, normalize and center on zero
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))

# build training model Here
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# note num samples = number of training samples * 3 images per line * 2 flipped images
history_object = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*6,
                    verbose=1,
                    nb_epoch=7)

model.save('model.h5')

# plot the training and validation loss byepoch
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.show()
