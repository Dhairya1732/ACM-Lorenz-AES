import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import random as rd

def pad_to_square(image):
    """Pad the image to make it square."""
    height, width, _ = image.shape
    if height == width:
        return image, height, width  # Already square

    size = max(height, width)
    padded_image = np.zeros((size, size, 3), dtype=image.dtype)
    padded_image[:height, :width] = image
    return padded_image, height, width

def crop_to_original(image, original_height, original_width):
    """Crop the image back to its original size."""
    return image[:original_height, :original_width]

def compute_lorenz_parameters(image):
    """
    Compute Lorenz Map initial parameters (x, y, z) for the input color image.
    Parameters are derived from the grayscale version for simplicity.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray / 255.0

    # Mean Gradient Magnitude (x)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    x = np.mean(gradient_magnitude)

    # Standard Deviation of Pixel Intensities (y)
    y = np.std(gray)

    # Skewness of Pixel Distribution (z)
    z = skew(gray.flatten())

    # Normalize x, y, z to ensure they are in a consistent range
    x = x / np.max(gradient_magnitude) if np.max(gradient_magnitude) != 0 else x
    y = y / np.max(gray) if np.max(gray) != 0 else y
    z = (z + 10) / 20  # Shift skewness to fit into [0, 1]

    # Calculate the correlation between adjacent pixels
    # Flatten the Sobel images for correlation calculation
    sobel_x_flat = sobel_x.flatten()
    sobel_y_flat = sobel_y.flatten()
    
    # Compute the Pearson correlation coefficient between the x and y gradients
    correlation = np.corrcoef(sobel_x_flat, sobel_y_flat)[0, 1]
    
    # Normalize the correlation to be between 0.001 and 0.1 for dt
    dt = (correlation + 1) / 2 * 0.099 + 0.001  # Scale to [0.001, 0.1]

    return x, y, z, dt

def arnold_cat_map(image, iterations, dx_sequence, dy_sequence):
    """Applies the Arnold Cat Map to scramble the image."""
    height, width, _ = image.shape
    scrambled_image = np.copy(image)

    for i in range(iterations):
        temp = np.zeros_like(scrambled_image)
        p = dx_sequence[i]
        q = dy_sequence[i]
        for x in range(height):
            for y in range(width):
                new_x = (x + (y*p)) % height
                new_y = ((x*q) + (((p*q)+1) * y)) % width
                temp[new_x, new_y] = scrambled_image[x, y]
        scrambled_image = temp
    return scrambled_image

def inverse_arnold_cat_map(image, iterations, dx_sequence, dy_sequence):
    height, width, _ = image.shape
    unscrambled_image = np.copy(image)
    
    for i in range(iterations):
        temp = np.zeros_like(unscrambled_image)
        p = dx_sequence[i]
        q = dy_sequence[i]
        for x in range(height):
            for y in range(width):
                new_x = ((((p*q)+1) * x) + (y*(-p))) % height
                new_y = ((x*(-q)) + y) % width
                temp[new_x, new_y] = unscrambled_image[x, y]
        unscrambled_image = temp
    return unscrambled_image

def lorenz_map(seed, length):
    """Generates Lorenz chaotic sequence values for a given seed."""
    sigma, rho, beta = 10.0, 35.0, 8.0 / 3.0
    x, y, z, dt = seed
    dx_sequence, dy_sequence, dz_sequence = [], [], []

    # Define capping limits for dx, dy, dz
    cap_value = 1e50

    for _ in range(length):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt

        # Cap the values to prevent extreme growth
        dx = min(dx, cap_value)
        dy = min(dy, cap_value)
        dz = min(dz, cap_value)

        # Update the Lorenz system state
        x += dx
        y += dy
        z += dz

        # Append chaotic values
        dx_sequence.append(abs(round(dx)))
        dy_sequence.append(abs(round(dy)))
        dz_sequence.append(abs(round(dz)))

    return dx_sequence, dy_sequence, dz_sequence

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    padded_image, original_height, original_width =  pad_to_square(image)

    # Parameters
    x, y, z, dt = compute_lorenz_parameters(image)
    seed = (x, y, z, dt)

    # Generate Lorenz sequences
    length = rd.randint(20,50)
    dx_sequence, dy_sequence, dz_sequence = lorenz_map(seed, length)
    acm_iterations = length

    # Scramble using ACM
    scrambled_image = arnold_cat_map(padded_image, acm_iterations, dx_sequence, dy_sequence)

    dx_sequence = list(reversed(dx_sequence))
    dy_sequence = list(reversed(dy_sequence))

    # Reverse ACM scrambling
    descrambled_image = inverse_arnold_cat_map(scrambled_image, acm_iterations, dx_sequence, dy_sequence) 

    cropped_image = crop_to_original(descrambled_image, original_height, original_width)

    # Display images
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(scrambled_image)
    plt.title("Encrypted Image")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cropped_image)
    plt.title("Decrypted Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run the program
main("test2.jpeg")