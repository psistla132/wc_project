import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, Add, LeakyReLU
from tensorflow.keras.models import Model
import scipy.io as sio 
import numpy as np
import math
import time

# Environment setting
envir = 'indoor' # 'indoor' or 'outdoor'
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height * img_width * img_channels
encoded_dim = 512  # compress rate=1/4->dim.=512

file = 'CsiNet_' + (envir) + '_dim' + str(encoded_dim)

# Function to recreate the model architecture
def create_model():
    inputs = Input(shape=(img_channels, img_height, img_width))
    
    # Encoder
    x = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    
    # Flatten and compress
    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim)(x)
    
    # Decoder
    x = Dense(img_total)(encoded)
    x = Reshape((img_channels, img_height, img_width))(x)
    
    # First residual block
    residual1 = x
    x = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Add()([residual1, x])
    x = LeakyReLU(alpha=0.3)(x)
    
    # Second residual block
    residual2 = x
    x = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Add()([residual2, x])
    x = LeakyReLU(alpha=0.3)(x)
    
    # Output layer
    outputs = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first', activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Create the model
autoencoder = create_model()

# Compile the model (needed before loading weights)
autoencoder.compile(optimizer='adam', loss='mse')

# Load weights - adjust the path as needed
weights_file = "model_%s.h5" % file
try:
    autoencoder.load_weights(weights_file)
    print(f"Successfully loaded weights from {weights_file}")
except:
    print(f"Failed to load weights from {weights_file}, using uninitialized model")

# Data loading
if envir == 'indoor':
    mat = sio.loadmat('DATA_Htestin.mat')
    x_test = mat['HT']  # array
elif envir == 'outdoor':
    mat = sio.loadmat('DATA_Htestout.mat')
    x_test = mat['HT']  # array

x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))

# Testing data
tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

# Calculating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('DATA_HtestFin_all.mat')
    X_test = mat['HF_all']  # array
elif envir == 'outdoor':
    mat = sio.loadmat('DATA_HtestFout_all.mat')
    X_test = mat['HF_all']  # array

X_test = np.reshape(X_test, (len(X_test), img_height, 125))
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]

n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))
n1 = n1.astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))
n2 = n2.astype('float64')
aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))
rho = np.mean(aa/(n1*n2), axis=1)
X_hat = np.reshape(X_hat, (len(X_hat), -1))
X_test = np.reshape(X_test, (len(X_test), -1))
power = np.sum(abs(x_test_C)**2, axis=1)
power_d = np.sum(abs(X_hat)**2, axis=1)
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))
print("Correlation is ", np.mean(rho))

# Visualization
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    
    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 + 1j*(x_hat[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

plt.show()