import cv2
import numpy as np
from scipy.stats import skew

def crop_to_original(image, original_height, original_width):
    """Crop the image back to its original size."""
    return image[:original_height, :original_width]

def pad_to_square(image):
    """Pad the image to make it square."""
    height, width, _ = image.shape
    if height == width:
        return image, height, width  # Already square

    size = max(height, width)
    padded_image = np.zeros((size, size, 3), dtype=image.dtype)
    padded_image[:height, :width] = image
    return padded_image, height, width

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