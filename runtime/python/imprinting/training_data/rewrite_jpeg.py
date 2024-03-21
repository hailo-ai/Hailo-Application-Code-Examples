from PIL import Image
import os, numpy as np

def print_image_mode(image_path):
    try:
        # Open the image
        img = Image.open(image_path).convert("RGB")
        matrix = np.array(img)
        new_img = Image.fromarray(matrix)
        print(type(matrix), matrix.shape)
        new_img.save(image_path, format='JPEG', quality=95)
        print('loaded and saved ', image_path)
    except Exception as e:
        print(f"Error: {e}")

# Replace 'your_image.jpg' with the path to your JPEG image
for image_path in os.listdir('.'):
#image_path = '0001.jpg'
    print_image_mode(image_path)
