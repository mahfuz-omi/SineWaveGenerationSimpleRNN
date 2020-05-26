import math
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM,Dense,SimpleRNN


x = np.arange(200)
print(x)
y = [math.sin(x) for x in x]
print(y)


plt.plot(x,y)
plt.show()

x_next = np.arange(100,300)
y_next = [math.sin(x) for x in x_next]

plt.plot(x_next,y_next)
plt.show()


MEMORY_LENGTH = 2
prev_data = []
next_data = []
for i in range(100 - MEMORY_LENGTH):
    char_list = []
    for char in y[i:i + MEMORY_LENGTH]:
        char_list.append(char)
    prev_data.append(char_list)
    next_data.append(y[i + MEMORY_LENGTH])

print('prev data:')
print(prev_data)

print('next data')
print(next_data)

# convert 2d data to 3d
# no need vector, so 1 in last dimension
X_train = np.array(prev_data).reshape(len(prev_data),MEMORY_LENGTH,1)
print("X_train:",X_train)

# it will remain 2d
y_train = np.array(next_data).reshape(len(next_data),1)

print('y_train',y_train)

# now build test data

prev_data = []
next_data = []
for i in range(100 - MEMORY_LENGTH):
    char_list = []
    for char in y_next[i:i + MEMORY_LENGTH]:
        char_list.append(char)
    prev_data.append(char_list)
    next_data.append(y_next[i + MEMORY_LENGTH])


X_test = np.array(prev_data).reshape(len(prev_data),MEMORY_LENGTH,1)
print("X_test:",X_test)

# it will remain 2d
y_test = np.array(next_data).reshape(len(next_data),1)

print('y_train',y_test)


def createModel():
    # Building the model
    # We use a single-layer LSTM model with 128 neurons,
    #  a fully connected layer, and a softmax function for activation.

    model = Sequential()
    # X has shape (len(prev_words),WORD_LENGTH,len(unique_words))
    # so, sample shape(shape[0] will not go here
    # By default, return_sequences=False.
    # If we want to add more LSTM layers,
    # then the last LSTM layer must add return_sequences=True
    model.add(SimpleRNN(units=32, input_shape=(MEMORY_LENGTH, 1),activation='relu'))

    # output class number
    # it is regression problem
    # so, it won't have any activation function
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, y_train, validation_split=0.05, batch_size=1, epochs=2, shuffle=True)

    # save this model
    model.save('keras_sine_wave_generation_model.h5')
    return model

def load_saved_model():
    from keras.models import load_model
    model = load_model('keras_sine_wave_generation_model.h5')
    return model


model = createModel()

# 2D data
input_data = X_train[-1]
# print(len(input_text))
print('input Text:')
print(input_data)

# input Text:
# [[ 0.37960774]
#  [-0.57338187]]

generatedDataSequences = []
def generateData(input2D,num_data=100):

    for i in range(0,num_data):
        data = generateSinglePoint(input2D)
        generatedDataSequences.append(data)

        # remove first character
        input2D[0] = input2D[1]
        input2D[1] = np.array(data)


    print("Generated data:")
    print(generatedDataSequences)


def generateSinglePoint(input2D):
    y_pred = model.predict(np.array([input2D]), verbose=0)[0]
    return y_pred[0]


generateData(input_data)

# print generated sine wave sequence
plt.plot(range(len(generatedDataSequences)),generatedDataSequences)
plt.show()