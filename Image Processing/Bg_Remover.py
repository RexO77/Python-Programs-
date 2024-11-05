from rembg import remove
from PIL import Image
import io

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
    input_path = 'D:\\Exam Stuff\\Nischal Skanda_CANDID.jpg'  # Path to the input image
    output_path = 'D:\\Exam Stuff\\Removed_BG.jpeg'  # Path to save the output image

    remove_background(input_path, output_path)
    print(f"Background removed and saved to {output_path}")