import numpy as np

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