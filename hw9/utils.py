from PIL import Image

def show(arr):
    new_arr = arr.copy()
    new_arr[arr==-1] = 0
    new_arr[arr==1] = 255
    Image.fromarray(new_arr).show()