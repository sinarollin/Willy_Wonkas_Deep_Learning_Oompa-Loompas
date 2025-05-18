#Authors: Timo Michoud, Sina Röllin, Veronika Podliesnova


#Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def plot_rgb_distribution(color_data):
    """
    Plots the RGB distribution of the detected colours from the reference pictures.

    Parameters
    ----------
    color_data : dict
        A dictionary containing color names and their corresponding RGB values.
    """

    _, ax = plt.subplots(figsize=(12, 8))
    x = 0
    yticks = []
    ylabels = []

    for name, colors in color_data.items():
        for color in colors:
            #normalize RGB values
            rgb_norm = tuple([c / 255 for c in color])
            ax.add_patch(mpatches.Rectangle((x, 0), 1, 1, color=rgb_norm))
            x += 1
        #draw a vertical line for each color set to distinguish them
        ax.axvline(x=x, color='white', linewidth=1.5, linestyle='-')
        yticks.append((x - len(colors) / 2))
        ylabels.append(name)

    ax.set_xlim(0, x)
    ax.set_ylim(0, 1)
    ax.set_xticks(yticks)
    ax.set_xticklabels(ylabels, rotation=90)
    ax.set_yticks([])
    ax.set_title("Color Distribution from Reference Pictures")

    plt.tight_layout()
    plt.show()





def rgb_to_hsv(rgb):
    """
    Converts an RGB color to HSV color space.
    
    Parameters
    ----------
    rgb : tuple
        A tuple of three integers representing the RGB color.
        
    Returns
    -------
    hsv : tuple
        A tuple of three integers representing the HSV color.
    """
    color = np.uint8([[rgb]])
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv[0][0]


# Function to create a combined mask in RGB
def create_combined_mask_rgb(image_rgb, color_data, tolerance=(30, 30, 30)):
    """
    Create a combined mask in RGB color space.

    Parameters
    ----------
    image_rgb: Input image in RGB format
    color_data: Dictionary of reference colors
    tolerance: Tolerance for color matching (R, G, B)

    Returns
    -------
    Combined mask
    """
    combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    for _, rgb_colors in color_data.items():
        for rgb in rgb_colors:
            # Define lower and upper bounds for the color with tolerance
            lower = np.array([max(c - t, 0) for c, t in zip(rgb, tolerance)], dtype=np.uint8)
            upper = np.array([min(c + t, 255) for c, t in zip(rgb, tolerance)], dtype=np.uint8)

            # Create a mask for the current color
            mask = cv2.inRange(image_rgb, lower, upper)

            # Combine the mask with the overall mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # # Apply morphological operations to fill small holes and remove noise
    kernel_fill = np.ones((30, 30), np.uint8)
    kernel_open = np.ones((25, 25), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_fill)  # Fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)   # Remove noise

    return combined_mask


def create_combined_mask_hsv(image_bgr, color_data, tolerance=(5, 40, 30), kernel_fill=20, kernel_open=20):
    """
    Creates a combined mask for the given image using the HSV color space.
    
    Parameters
    ----------
    image_bgr : numpy.ndarray
        The input image in BGR format.
    color_data : dict
        A dictionary containing color names and their corresponding RGB values.
    tolerance : tuple
        A tuple of three integers representing the tolerance for hue, saturation, and value.
    kernel_fill : int
        The size of the kernel for morphological closing operation.
    kernel_open : int
        The size of the kernel for morphological opening operation.
    Returns
    -------
    combined_mask : numpy.ndarray
        The combined mask of the detected colors.
    """
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for reference_name, rgb_colors in color_data.items():
        for rgb in rgb_colors:
            hsv = rgb_to_hsv(rgb)
            lower = np.array([
                max(hsv[0] - tolerance[0], 0),
                max(hsv[1] - tolerance[1], 0),
                max(hsv[2] - tolerance[2], 0)
            ], dtype=np.uint8)  # Explicitly cast to uint8
            upper = np.array([
                min(hsv[0] + tolerance[0], 255),
                min(hsv[1] + tolerance[1], 255),
                min(hsv[2] + tolerance[2], 255)
            ], dtype=np.uint8)  # Explicitly cast to uint8
            mask = cv2.inRange(hsv_image, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Apply morphological operations
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_fill = np.ones((kernel_fill, kernel_fill), np.uint8)
    kernel_open = np.ones((kernel_open, kernel_open), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_fill)  # Fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)   # Remove noise


    return combined_mask


# Function to create a combined mask in RGB
def create_combined_mask_rgb(image_rgb, color_data, tolerance=(30, 30, 30)):
    """
    Create a combined mask in RGB color space.

    Parameters
    ----------
    image_rgb: Input image in RGB format
    color_data: Dictionary of reference colors
    tolerance: Tolerance for color matching (R, G, B)

    Returns
    -------
    return: Combined mask
    """
    combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    for _, rgb_colors in color_data.items():
        for rgb in rgb_colors:
            # Define lower and upper bounds for the color with tolerance
            lower = np.array([max(c - t, 0) for c, t in zip(rgb, tolerance)], dtype=np.uint8)
            upper = np.array([min(c + t, 255) for c, t in zip(rgb, tolerance)], dtype=np.uint8)

            # Create a mask for the current color
            mask = cv2.inRange(image_rgb, lower, upper)

            # Combine the mask with the overall mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # # Apply morphological operations to fill small holes and remove noise
    kernel_fill = np.ones((30, 30), np.uint8)
    kernel_open = np.ones((25, 25), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_fill)  # Fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)   # Remove noise

    return combined_mask