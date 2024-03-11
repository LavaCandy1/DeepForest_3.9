from PIL import Image
import numpy as np

img = Image.open('your_image.jpg').convert('L')

print(img)