from rembg import remove
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import io

def remove_background(input_path):
    # Open the input image
    with open(input_path, 'rb') as input_file:
        input_image = input_file.read()
    
    # Remove the background with alpha matting enabled
    output_image = remove(
        input_image,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )
    return output_image

def upload_image():
    global input_img, input_image_path
    input_image_path = filedialog.askopenfilename(
        title="Select Input Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if input_image_path:
        img = Image.open(input_image_path)
        img.thumbnail((300, 300))
        input_img = ImageTk.PhotoImage(img)
        input_label.config(image=input_img)
        input_label.image = input_img
        status_label.config(text="Image loaded.")

def process_image():
    if not input_image_path:
        messagebox.showerror("Error", "No input file selected.")
        return

    status_label.config(text="Processing, please wait...")
    root.update_idletasks()

    # Run the processing in a separate thread
    threading.Thread(target=run_remove).start()

def run_remove():
    global output_img
    try:
        output_image_data = remove_background(input_image_path)

        # Load the output image from bytes
        output_image = Image.open(io.BytesIO(output_image_data))
        output_image.thumbnail((300, 300))
        output_img = ImageTk.PhotoImage(output_image)

        # Update the output label in the main thread
        output_label.config(image=output_img)
        output_label.image = output_img
        status_label.config(text="Processing completed.")
    except Exception as e:
        status_label.config(text="An error occurred.")
        messagebox.showerror("Error", str(e))

def save_image():
    if not output_label.image:
        messagebox.showerror("Error", "No processed image to save.")
        return

    output_path = filedialog.asksaveasfilename(
        title="Save Output Image As",
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")]
    )
    if not output_path:
        return

    try:
        # Save the output image
        output_image_data = output_label.image._PhotoImage__photo.write(output_path)
        messagebox.showinfo("Success", f"Image saved to {output_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Initialize the main window
root = tk.Tk()
root.title("Advanced Background Remover")

# Variables to hold image data
input_image_path = None
input_img = None
output_img = None

# Frames for input and output images
input_frame = tk.LabelFrame(root, text="Original Image")
input_frame.grid(row=0, column=0, padx=10, pady=10)

output_frame = tk.LabelFrame(root, text="Processed Image")
output_frame.grid(row=0, column=1, padx=10, pady=10)

# Labels to display images
input_label = tk.Label(input_frame)
input_label.pack()

output_label = tk.Label(output_frame)
output_label.pack()

# Frame for buttons
button_frame = tk.Frame(root)
button_frame.grid(row=1, column=0, columnspan=2, pady=10)

# Buttons
upload_button = tk.Button(button_frame, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=0, padx=5)

process_button = tk.Button(button_frame, text="Remove Background", command=process_image)
process_button.grid(row=0, column=1, padx=5)

save_button = tk.Button(button_frame, text="Save Image", command=save_image)
save_button.grid(row=0, column=2, padx=5)

# Status label
status_label = tk.Label(root, text="Welcome to the Background Remover")
status_label.grid(row=2, column=0, columnspan=2, pady=5)

# Start the main event loop
root.mainloop()