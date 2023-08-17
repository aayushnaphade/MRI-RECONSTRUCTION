import os
import numpy as np
from tensorflow import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, Lambda , concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
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

# Complex to real and imaginary parts conversion
def complex_to_real_imag(x):
    real = tf.math.real(x)
    imag = tf.math.imag(x)
    return real, imag

# Real and imaginary parts to complex conversion
def real_imag_to_complex(x):
    return tf.complex(x[0], x[1])

# Calculate SNR
def calculate_snr(original, reconstructed):
    signal_power = np.sum(np.abs(original)**2)
    noise_power = np.sum(np.abs(original - reconstructed)**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Calculate RMSE
def calculate_rmse(original, reconstructed):
    return np.sqrt(np.mean((np.abs(original) - np.abs(reconstructed))**2))

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
# Set the undersampling ratio
undersampling_ratio = 0.5

# Perform 2D Fourier transform on the training data
x_train_fourier = np.fft.fft2(x_train)

# Calculate the number of frequencies to keep in the undersampled Fourier data
num_frequencies_to_keep = int(undersampling_ratio * x_train.size)

# Get the indices of all frequencies in the Fourier data
all_indices = np.arange(x_train.size)

# Randomly select the indices of the frequencies to keep
undersampled_indices = np.random.choice(all_indices, size=num_frequencies_to_keep, replace=False)

# Create a new undersampled Fourier data by setting the frequencies not in undersampled_indices to zero
undersampled_fourier = np.zeros_like(x_train_fourier)
undersampled_fourier.flat[undersampled_indices] = x_train_fourier.flat[undersampled_indices]


# Assuming you have a validation data array named "x_val"
# Perform 2D Fourier transform on the validation data
x_val_fourier = np.fft.fft2(x_val)

# Calculate the number of frequencies to keep in the undersampled Fourier data for validation data
num_frequencies_to_keep_val = int(undersampling_ratio * x_val.size)

# Get the indices of all frequencies in the Fourier data for validation data
all_indices_val = np.arange(x_val.size)

# Randomly select the indices of the frequencies to keep for validation data
undersampled_indices_val = np.random.choice(all_indices_val, size=num_frequencies_to_keep_val, replace=False)

# Create a new undersampled Fourier data for validation data
undersampled_fourier_val = np.zeros_like(x_val_fourier)
undersampled_fourier_val.flat[undersampled_indices_val] = x_val_fourier.flat[undersampled_indices_val]

# Perform inverse Fourier transform to get the undersampled validation data
x_val_undersampled = np.abs(np.fft.ifft2(undersampled_fourier_val))


# Perform inverse Fourier transform to get the undersampled training data
x_train_undersampled = np.abs(np.fft.ifft2(undersampled_fourier))
# Define U-Net model
def create_unet(input_shape):
    inputs = Input(shape=input_shape, dtype=tf.complex64)
    
    # Split complex input into real and imaginary parts
    real_part = Lambda(lambda x: tf.math.real(x))(inputs)
    imag_part = Lambda(lambda x: tf.math.imag(x))(inputs)
    
    # Encoding path
    conv1_real = Conv2D(32, (3, 3), activation='relu', padding='same')(real_part)
    conv2_real = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_real)
    
    conv1_imag = Conv2D(32, (3, 3), activation='relu', padding='same')(imag_part)
    conv2_imag = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_imag)

    # Bottleneck
    merged_conv = concatenate([conv2_real, conv2_imag], axis=-1)
    bottleneck = Conv2D(128, (3, 3), activation='relu', padding='same')(merged_conv)

    # Decoding path and skip connections
    deconv2_real = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(bottleneck)
    skip2_real = concatenate([deconv2_real, conv1_real], axis=-1)
    deconv3_real = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(skip2_real)
    
    deconv2_imag = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(bottleneck)
    skip2_imag = concatenate([deconv2_imag, conv1_imag], axis=-1)
    deconv3_imag = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(skip2_imag)

    # Real-valued transposed convolution layer
    deconv1_real = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(deconv3_real)
    
    # Complex-valued output layer
    outputs = Conv2D(1, (3, 3), activation='linear', padding='same')(deconv1_real)
    
    # Merge real and imaginary parts back to complex-valued data
    outputs_complex = Lambda(real_imag_to_complex)([outputs, tf.zeros_like(outputs)])
    
    model = Model(inputs=inputs, outputs=outputs_complex)
    
    return model

# Create U-Net model
input_shape = (256, 256, 1)
unet_model = create_unet(input_shape)

# Compile the model
unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
# Train the U-Net model
batch_size = 16
epochs = 1

# Train the model
history = unet_model.fit(x_train_undersampled, x_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val_undersampled, x_val))

# Test the U-Net model
x_test_fourier = np.fft.fft2(x_test)

# Calculate the number of frequencies to keep in the undersampled Fourier data for test data
num_frequencies_to_keep_test = int(undersampling_ratio * x_test.size)

# Get the indices of all frequencies in the Fourier data for test data
all_indices_test = np.arange(x_test.size)

# Randomly select the indices of the frequencies to keep for test data
undersampled_indices_test = np.random.choice(all_indices_test, size=num_frequencies_to_keep_test, replace=False)

# Create a new undersampled Fourier data for test data
undersampled_fourier_test = np.zeros_like(x_test_fourier)
undersampled_fourier_test.flat[undersampled_indices_test] = x_test_fourier.flat[undersampled_indices_test]

# Perform inverse Fourier transform to get the undersampled test data
x_test_undersampled = np.abs(np.fft.ifft2(undersampled_fourier_test))

# Test the U-Net model on the test data
reconstructed_images = unet_model.predict(x_test_undersampled)

# Calculate SNR (Signal-to-Noise Ratio) and RMSE (Root Mean Squared Error)
snr_values = []
rmse_values = []
for i in range(x_test.shape[0]):
    original_image = x_test[i]
    reconstructed_image = reconstructed_images[i]
    
    # Calculate SNR
    snr = calculate_snr(original_image, reconstructed_image)
    snr_values.append(snr)
    
    # Calculate RMSE
    rmse = calculate_rmse(original_image, reconstructed_image)
    rmse_values.append(rmse)

# Print average SNR and RMSE values
avg_snr = np.mean(snr_values)
avg_rmse = np.mean(rmse_values)
print(f'Average SNR: {avg_snr:.2f}')
print(f'Average RMSE: {avg_rmse:.2f}')

# Visualize a few original, undersampled, and reconstructed images
num_visualize = 5
for i in range(num_visualize):
    plt.figure(figsize=(18, 6))
    
    # Visualize the undersampled image
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(x_test_undersampled[i]), cmap='gray')
    plt.title('Input Undersampled Image')
    
    # Visualize the original image
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(np.abs(x_test[i])), cmap='gray')
    plt.title('Original Image (Magnitude)')
    
    # Visualize the reconstructed image
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(np.abs(reconstructed_images[i])), cmap='gray')
    plt.title('Reconstructed Image (Magnitude)')
    
    plt.show()
