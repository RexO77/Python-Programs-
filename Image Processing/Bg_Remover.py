from rembg import remove
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

def remove_background(input_path, output_path):
    # Open the input image
    with open(input_path, 'rb') as input_file:
        input_image = input_file.read()

    # Remove the background
    output_image = remove(input_image)

    # Save the output image without compression
    with open(output_path, 'wb') as output_file:
        output_file.write(output_image)

def upload_image():
    global input_path
    input_path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if input_path:
        img = Image.open(input_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        input_label.config(image=img)
        input_label.image = img
        input_label.pack()

def save_image():
    if not input_path:
        messagebox.showerror("Error", "No input file selected.")
        return

    output_path = filedialog.asksaveasfilename(title="Save Output Image As", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")])
    if not output_path:
        messagebox.showerror("Error", "No output file selected.")
        return

    remove_background(input_path, output_path)
    messagebox.showinfo("Success", f"Background removed and saved to {output_path}")

# Create the main window
root = tk.Tk()
root.title("Background Remover")

# Create a frame for the input image
input_frame = tk.Frame(root)
input_frame.pack(pady=10)
input_label = tk.Label(input_frame, text="Upload an image")
input_label.pack()

# Create buttons for uploading and saving images
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=5)

save_button = tk.Button(root, text="Save Image", command=save_image)
save_button.pack(pady=5)

# Run the application
root.mainloop()