from rembg import remove
from PIL import Image
import io
import tkinter as tk
from tkinter import filedialog

def remove_background(input_path, output_path):
    # Open the input image
    with open(input_path, 'rb') as input_file:
        input_image = input_file.read()

    # Remove the background
    output_image = remove(input_image)

    # Save the output image without compression
    with open(output_path, 'wb') as output_file:
        output_file.write(output_image)

if __name__ == "__main__":
    # Create a Tkinter root window (it will not be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select the input image
    input_path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not input_path:
        print("No input file selected. Exiting.")
        exit()

    # Open a file dialog to select the output image path
    output_path = filedialog.asksaveasfilename(title="Save Output Image As", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")])
    if not output_path:
        print("No output file selected. Exiting.")
        exit()

    remove_background(input_path, output_path)
    print(f"Background removed and saved to {output_path}")