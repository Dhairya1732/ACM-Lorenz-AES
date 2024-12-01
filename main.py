import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
from  utils import crop_to_original, pad_to_square, compute_lorenz_parameters
from ACM_Lorenz import lorenz_map, arnold_cat_map, inverse_arnold_cat_map

def pad_image_to_block_size(image, block_size=16):
    """
    Pads the image to make its byte array length a multiple of the AES block size.
    """
    h, w, c = image.shape
    padded_h = (h + block_size - 1) // block_size * block_size
    padded_w = (w + block_size - 1) // block_size * block_size
    padded_image = np.zeros((padded_h, padded_w, c), dtype=np.uint8)
    padded_image[:h, :w, :] = image
    return padded_image

def aes_encrypt_image(image, key, iv):
    """
    Encrypts an image using AES (CBC mode) while preserving its structure.
    Each channel is encrypted independently.
    """
    h, w, c = image.shape
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Encrypt each channel independently
    encrypted_channels = []
    for i in range(c):
        # Flatten the channel and pad to a multiple of block size
        flat_channel = image[:, :, i].flatten()
        padding_length = 16 - (len(flat_channel) % 16)
        flat_channel = np.pad(flat_channel, (0, padding_length), mode='constant', constant_values=0)
        
        # Encrypt the flattened channel
        encrypted_data = cipher.encrypt(flat_channel.tobytes())
        
        # Convert back to array
        encrypted_channel = np.frombuffer(encrypted_data, dtype=np.uint8)[:h * w]
        encrypted_channels.append(encrypted_channel.reshape(h, w))
    
    # Stack the channels back
    encrypted_image = np.stack(encrypted_channels, axis=-1)
    return encrypted_image

def aes_decrypt_image(encrypted_image, key, iv):
    """
    Decrypts an AES-encrypted image channel by channel.
    """
    h, w, c = encrypted_image.shape
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt each channel independently
    decrypted_channels = []
    for i in range(c):
        # Flatten the encrypted channel
        flat_channel = encrypted_image[:, :, i].flatten()
        
        # Decrypt the flattened channel
        decrypted_data = cipher.decrypt(flat_channel.tobytes())
        
        # Remove padding (if any)
        decrypted_channel = np.frombuffer(decrypted_data, dtype=np.uint8)[:h * w]
        decrypted_channels.append(decrypted_channel.reshape(h, w))
    
    # Stack the channels back
    decrypted_image = np.stack(decrypted_channels, axis=-1)
    return decrypted_image

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    padded_image, original_height, original_width =  pad_to_square(image)

    # Generate a random AES key and IV (for example purposes)
    key = os.urandom(16)  
    iv = os.urandom(16)  

    # Parameters
    x, y, z, dt = compute_lorenz_parameters(image)
    seed = (x, y, z, dt)

    # Generate Lorenz sequences
    length = rd.randint(20,50)
    dx_sequence, dy_sequence, dz_sequence = lorenz_map(seed, length)
    acm_iterations = length

    # Encrypt the image using AES before scrambling
    encrypted_image1 = aes_encrypt_image(padded_image, key, iv)

    # Scramble using ACM
    scrambled_image = arnold_cat_map(encrypted_image1, acm_iterations, dx_sequence, dy_sequence)

    # Encrypt the image using AES after scrambling
    encrypted_image2 = aes_encrypt_image(scrambled_image, key, iv)

    dx_sequence.reverse()
    dy_sequence.reverse()

    # Decrypt the image before unscrambling
    decrypted_image1 = aes_decrypt_image(encrypted_image2, key, iv)

    # Reverse ACM scrambling
    descrambled_image = inverse_arnold_cat_map(decrypted_image1, acm_iterations, dx_sequence, dy_sequence) 

    # Decrypt the image after unscrambling
    decrypted_image2 = aes_decrypt_image(descrambled_image, key, iv)

    cropped_image = crop_to_original(decrypted_image2, original_height, original_width)

    # Display images
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(encrypted_image1)
    plt.title("Encrypted Image 1")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(scrambled_image)
    plt.title("Scrambled Image")
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(encrypted_image2)
    plt.title("Encrypted Image 2")
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(decrypted_image1)
    plt.title("Decrypted Image 1")
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(descrambled_image)
    plt.title("Descrambled Image")
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(decrypted_image2)
    plt.title("Decrypted Image 2")
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(cropped_image)
    plt.title("Cropped Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run the program
main("test2.jpeg")