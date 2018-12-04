from PIL import Image
import matplotlib.pyplot as plt

def show(arr):
    new_arr = arr.copy()
    new_arr[arr==-1] = 0
    new_arr[arr==1] = 255
    plt.figure()
    plt.imshow(new_arr, cmap='gray')
    # Image.fromarray(new_arr).show()

def save(arr, path):
    new_arr = arr.copy()
    new_arr[arr == -1] = 0
    new_arr[arr == 1] = 255
    Image.fromarray(new_arr).convert('L').save('{}.png'.format(path))