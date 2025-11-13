from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Build a simple CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Dummy training data
X_train = np.random.rand(10, 48, 48, 1)
y_train = np.eye(7)[np.random.randint(0, 7, 10)]

# Train briefly
model.fit(X_train, y_train, epochs=1)

# Save the model
model.save('trained_models/model.h5')
