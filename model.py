
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # Assuming 7 emotions

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # Load dataset (You need to implement your dataset loading)
    # Example: data = pd.read_csv('path_to_your_dataset.csv')
    
    # Data preprocessing steps here...
    
    model = create_model()
    
    # Fit the model (you need to replace train_data and train_labels)
    # model.fit(train_data, train_labels, epochs=10, validation_split=0.1)
    
    model.save('trained_models/model.h5')

if __name__ =="__main__":
    train_model()

