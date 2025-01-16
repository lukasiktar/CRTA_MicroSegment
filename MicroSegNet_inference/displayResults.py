import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_results(number_of_images_to_show, output_dir):
    """
    Display inference results.

    Args:
        number_of_images_to_show - the number of images to show
        output_dir - the output directory with inference results.
    """

    #Store the image paths 
    output_images = [os.path.join(output_dir, file_path) for file_path in os.listdir(output_dir) if file_path.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]


     # Create the plot
    cols = 5  # Define number of columns
    rows = (number_of_images_to_show + cols - 1) // cols  # Calculate rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(25, 25))

    # Flatten the axes array
    #axes = axes.ravel()

    # Plot the images
    for i, ax in enumerate(axes):
        if i < number_of_images_to_show:
            image_path = output_images[i]
            image = mpimg.imread(image_path)  # Load the image
            ax.imshow(image, cmap='gray')  # Display the image
            ax.axis('off')  # Hide axis
            ax.set_title(f"Image {i+1}")  # Add title
        else:
            ax.axis('off')  # Hide any unused subplot

    fig.suptitle("Prostate Segmentation Results", fontsize=20)
    plt.tight_layout()
    plt.show()
