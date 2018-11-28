from PIL import Image

def show(arr):
    arr[arr==-1] = 0
    arr[arr==1] = 255
    Image.fromarray(arr).show()