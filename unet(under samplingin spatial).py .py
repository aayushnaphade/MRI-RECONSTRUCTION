import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Lambda

import scipy.io as sio
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load and preprocess the data
def load_and_preprocess_data(file_names, data_path):
    data_list = []
    for file_name in file_names:
        mat_contents = sio.loadmat(os.path.join(data_path, file_name))
        variable_name = file_name.split('.')[0]
        real_values = mat_contents[variable_name][:, :, 0]
        imag_values = mat_contents[variable_name][:, :, 1]
        complex_data = real_values + 1j * imag_values
        magnitude_data = np.abs(complex_data)
        magnitude_data = np.expand_dims(magnitude_data, axis=-1)
        data_list.append(magnitude_data)
    return np.array(data_list)

def complex_to_real_imag(x):
    real = tf.math.real(x)
    imag = tf.math.imag(x)
    return real, imag

def real_imag_to_complex(x):
    return tf.complex(x[0], x[1])

def create_unet(input_shape):
    inputs = Input(shape=input_shape, dtype=tf.complex64)
    real_part, imag_part = Lambda(complex_to_real_imag)(inputs)
    conv1_real = Conv2D(32, (3, 3), activation='relu', padding='same')(real_part)
    conv2_real = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_real)
    conv1_imag = Conv2D(32, (3, 3), activation='relu', padding='same')(imag_part)
    conv2_imag = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_imag)
    merged_conv = concatenate([conv2_real, conv2_imag], axis=-1)
    deconv1_real = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(merged_conv)
    outputs = Conv2D(1, (3, 3), activation='linear', padding='same')(deconv1_real)
    outputs_complex = Lambda(real_imag_to_complex)([outputs, tf.zeros_like(outputs)])
    model = Model(inputs=inputs, outputs=outputs_complex)
    return model

# Perform undersampling on the data
def perform_undersampling(data, undersampling_ratio):
    undersampled_data = []
    for sample in data:
        num_frequencies = int(sample.size * undersampling_ratio)
        undersampled_indices = np.random.choice(sample.size, size=num_frequencies, replace=False)
        undersampled_sample = np.zeros_like(sample)
        undersampled_sample.flat[undersampled_indices] = sample.flat[undersampled_indices]
        undersampled_data.append(undersampled_sample)
    return np.array(undersampled_data)



# Calculate average SNR
def calculate_avg_snr(original_data, reconstructed_data):
    snr_values = []
    for i in range(original_data.shape[0]):
        snr = calculate_snr(original_data[i], reconstructed_data[i])
        snr_values.append(snr)
    return np.mean(snr_values)

# Calculate average RMSE
def calculate_avg_rmse(original_data, reconstructed_data):
    rmse_values = []
    for i in range(original_data.shape[0]):
        rmse = calculate_rmse(original_data[i], reconstructed_data[i])
        rmse_values.append(rmse)
    return np.mean(rmse_values)

def calculate_snr(original, reconstructed):
    signal_power = np.sum(np.abs(original)**2)
    noise_power = np.sum(np.abs(original - reconstructed)**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_rmse(original, reconstructed):
    return np.sqrt(np.mean((np.abs(original) - np.abs(reconstructed))**2))


# Visualize original, undersampled, and reconstructed images
def visualize_results(undersampled_data, original_data, reconstructed_data):
    num_visualize = 5
    for i in range(num_visualize):
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.squeeze(undersampled_data[i]), cmap='gray')
        plt.title('Input Undersampled Image')
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(np.abs(original_data[i])), cmap='gray')
        plt.title('Original Image (Magnitude)')
        
        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(np.abs(reconstructed_data[i])), cmap='gray')
        plt.title('Reconstructed Image (Magnitude)')
        
        plt.show()


# Data paths and file names
data_path = 'actual/'
file_names = [f'MR{i}.mat' for i in range(1, 113)]

# Load and preprocess the data
data_array = load_and_preprocess_data(file_names, data_path)

# Split data into train, validation, and test sets
x_train, x_test = train_test_split(data_array, test_size=0.2, random_state=42)

# Set the undersampling ratio
undersampling_ratio = 0.5

# Perform undersampling on the training data
x_train_undersampled = perform_undersampling(x_train, undersampling_ratio)

# Create U-Net model
input_shape = (256, 256, 1)
unet_model = create_unet(input_shape)

# Compile the model
unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

# Train the model
unet_model.fit(x_train_undersampled, x_train, batch_size=16, epochs=1, validation_split=0.1)

# Perform undersampling on the test data
x_test_undersampled = perform_undersampling(x_test, undersampling_ratio)

# Reconstruct images using the trained model
reconstructed_images = unet_model.predict(x_test_undersampled)

# Evaluate the results
avg_snr = calculate_avg_snr(x_test, reconstructed_images)
avg_rmse = calculate_avg_rmse(x_test, reconstructed_images)
print(f'Average SNR: {avg_snr:.2f}')
print(f'Average RMSE: {avg_rmse:.2f}')

# Visualize a few original, undersampled, and reconstructed images
visualize_results(x_test_undersampled, x_test, reconstructed_images)
