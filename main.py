import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
from  utils import crop_to_original, pad_to_square, compute_lorenz_parameters
from ACM_Lorenz import lorenz_map, arnold_cat_map, inverse_arnold_cat_map

# Function to encrypt the image using AES
def aes_encrypt_image(image, key, iv):
    """Encrypt the image using AES encryption block-by-block."""
    height, width, channels = image.shape
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    # Create a new array for the encrypted image
    encrypted_image = np.zeros_like(image)
    
    # Iterate over the image in blocks of 16x16 pixels (AES block size)
    for channel in range(channels):  # Iterate over RGB channels separately
        for i in range(0, height, 16):
            for j in range(0, width, 16):
                block = image[i:i+16, j:j+16, channel]  # Get a 16x16 block for this channel
                
                # If block is smaller than 16x16, pad it
                if block.shape[0] < 16 or block.shape[1] < 16:
                    block = np.pad(block, ((0, 16 - block.shape[0]), (0, 16 - block.shape[1])), mode='constant')

                block_bytes = block.tobytes()
                encrypted_block = cipher.encrypt(pad(block_bytes, AES.block_size))  # Encrypt the block
                
                # Convert the encrypted block back to a numpy array and store it in the encrypted image
                encrypted_image[i:i+16, j:j+16, channel] = np.frombuffer(encrypted_block, dtype=np.uint8).reshape((16, 16))
    
    return encrypted_image

# Function to decrypt the image using AES
def aes_decrypt_image(encrypted_image, key, iv, original_shape):
    """Decrypt the image using AES decryption block-by-block."""
    height, width, channels = original_shape
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    # Create a new array for the decrypted image
    decrypted_image = np.zeros_like(encrypted_image)
    
    # Iterate over the image in blocks of 16x16 pixels (AES block size)
    for channel in range(channels):  # Iterate over RGB channels separately
        for i in range(0, height, 16):
            for j in range(0, width, 16):
                encrypted_block = encrypted_image[i:i+16, j:j+16, channel]  # Get a 16x16 encrypted block for this channel
                encrypted_block_bytes = encrypted_block.tobytes()
                
                # Decrypt the block and remove padding
                decrypted_block = unpad(cipher.decrypt(encrypted_block_bytes), AES.block_size)
                
                # Convert the decrypted block back to a numpy array and store it in the decrypted image
                decrypted_image[i:i+16, j:j+16, channel] = np.frombuffer(decrypted_block, dtype=np.uint8).reshape((16, 16))
    
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