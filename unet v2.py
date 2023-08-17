import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate , Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from matplotlib import pyplot as plt
import scipy.io as sio

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

# Data paths and file names
data_path = 'actual/'
file_names = [f'MR{i}.mat' for i in range(1, 113)]

# Load and preprocess the data
data_array = load_and_preprocess_data(file_names, data_path)

# Split data into train, validation, and test sets
def train_test_val_split(data_array, val_split=0.1, test_split=0.1):
    num_samples = data_array.shape[0]
    num_val = int(num_samples * val_split)
    num_test = int(num_samples * test_split)
    num_train = num_samples - num_val - num_test

    x_train = data_array[:num_train]
    x_val = data_array[num_train:num_train+num_val]
    x_test = data_array[num_train+num_val:]

    return x_train, x_val, x_test

x_train, x_val, x_test = train_test_val_split(data_array)

# Set the undersampling ratio (e.g., 0.5 means we keep 50% of the frequencies)
undersampling_ratio = 0.8

# Perform undersampling on the training data
def perform_undersampling(data):
    x_fourier = np.fft.fft2(data)
    num_frequencies_to_keep = int(undersampling_ratio * x_fourier.size)
    all_indices = np.arange(x_fourier.size)
    undersampled_indices = np.random.choice(all_indices, size=num_frequencies_to_keep, replace=False)
    undersampled_fourier = np.zeros_like(x_fourier)
    undersampled_fourier.flat[undersampled_indices] = x_fourier.flat[undersampled_indices]
    x_undersampled = np.abs(np.fft.ifft2(undersampled_fourier))
    return x_undersampled

x_train_undersampled = perform_undersampling(x_train)

# Create U-Net model
# def create_unet(input_shape):
#     inputs = Input(shape=input_shape, dtype=tf.complex64)
#     real_part, imag_part = tf.split(inputs, 2, axis=-1)
#     conv1_real = Conv2D(32, (3, 3), activation='relu', padding='same')(real_part)
#     conv2_real = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_real)
#     conv1_imag = Conv2D(32, (3, 3), activation='relu', padding='same')(imag_part)
#     conv2_imag = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_imag)
#     merged_conv = concatenate([conv2_real, conv2_imag], axis=-1)
#     deconv1_real = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(merged_conv)
#     outputs = Conv2D(1, (3, 3), activation='linear', padding='same')(deconv1_real)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
# Create U-Net model
def create_unet(input_shape):
    inputs = Input(shape=input_shape, dtype=tf.complex64)
    real_part = tf.math.real(inputs)
    imag_part = tf.math.imag(inputs)
    
    conv1_real = Conv2D(32, (3, 3), activation='relu', padding='same')(real_part)
    conv2_real = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_real)
    
    conv1_imag = Conv2D(32, (3, 3), activation='relu', padding='same')(imag_part)
    conv2_imag = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_imag)
    
    merged_conv = concatenate([conv2_real, conv2_imag], axis=-1)
    deconv1_real = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(merged_conv)
    
    outputs = Conv2D(1, (3, 3), activation='linear', padding='same')(deconv1_real)
    
    outputs_complex = tf.complex(outputs, tf.zeros_like(outputs))
    model = Model(inputs=inputs, outputs=outputs_complex)
    
    return model

# ... (rest of the code remains unchanged)


input_shape = (256, 256, 1)
unet_model = create_unet(input_shape)
unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

# Train the U-Net model
unet_model.fit(x_train_undersampled, x_train, batch_size=16, epochs=6, validation_data=(x_val, x_val))

# Test the model on the test data
x_test_undersampled = perform_undersampling(x_test)
reconstructed_images = unet_model.predict(x_test_undersampled)

# Calculate SNR and RMSE
def calculate_snr(original, reconstructed):
    signal_power = np.sum(np.abs(original)**2)
    noise_power = np.sum(np.abs(original - reconstructed)**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_rmse(original, reconstructed):
    return np.sqrt(np.mean((np.abs(original) - np.abs(reconstructed))**2))

snr_values = []
rmse_values = []
for i in range(x_test.shape[0]):
    original_image = x_test[i]
    reconstructed_image = reconstructed_images[i]
    snr = calculate_snr(original_image, reconstructed_image)
    rmse = calculate_rmse(original_image, reconstructed_image)
    snr_values.append(snr)
    rmse_values.append(rmse)

avg_snr = np.mean(snr_values)
avg_rmse = np.mean(rmse_values)
print(f'Average SNR: {avg_snr:.2f}')
print(f'Average RMSE: {avg_rmse:.2f}')

# Visualize results
num_visualize = 5
for i in range(num_visualize):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(x_test_undersampled[i]), cmap='gray')
    plt.title('Input Undersampled Image')
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(np.abs(x_test[i])), cmap='gray')
    plt.title('Original Image (Magnitude)')
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(np.abs(reconstructed_images[i])), cmap='gray')
    plt.title('Reconstructed Image (Magnitude)')
    plt.show()
