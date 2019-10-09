import numpy as np
from PIL import Image
import os
import glob
from skimage.measure import block_reduce




__myos__="win"

a=np.random.rand(30)
print(a)
b=np.arange(18).reshape(3,6)
print(b)

# Parameters of the images
DIM_1 = 192
DIM_2 = 168
# Dense Noise
Epsilon = 1000
CLASSES = 38
SAMPLE_EACH_CLASS = 58
DOWNSAMPLE_COEFFICIENT = 2 

def LoadImage():
    prg1Path=os.path.split(os.path.realpath(__file__))[0]
    print(prg1Path)
    # Current Path
    if __myos__=="win":
        currPath=prg1Path+"\\data\\CroppedYale\\"
    else:
        currPath = prg1Path+'/data/CroppedYale/'
    print(currPath)
    # Load the first image
    X_train= []

    os.chdir(currPath)
    classDirectory = glob.glob("yale*")

    # Record the image labels
    delta = [[0 for n in range(SAMPLE_EACH_CLASS*CLASSES)] for m in range(CLASSES)]

    pos = 0
    # Load images from different classes
    for i in range(CLASSES):
        # List all the class directories
        filePath = currPath + classDirectory[i]
        # print(os.getcwd())
        os.chdir(filePath)
        fileList = glob.glob("*.pgm")
        # Class i
        # Exculde 
        
        for file_item in range(SAMPLE_EACH_CLASS):
            img = Image.open(filePath+'/'+fileList[file_item])
            # print(type(img))
            img = block_reduce(np.array(img), block_size=(DOWNSAMPLE_COEFFICIENT, DOWNSAMPLE_COEFFICIENT), func=np.mean)
            if img.shape[0]!=240 :
	            # plt.imshow(img, cmap=plt.get_cmap('gray'))
	            # plt.gca().axis('off')
	            # plt.gcf().set_size_inches((5, 5))
	            # plt.show()
	            # print(file_item)
	            # print(img.shape)
	            X_train.append(np.ndarray.flatten((np.array(img))))
	            delta[i][pos] = 1
	            pos += 1
    print("Delta, shape:", np.array(delta).shape)
    print("X_train, shape", np.array(X_train).shape) 
    return np.array(X_train).T, np.array(delta)

if __name__=="__main__":
    X_train,delta=LoadImage()
    print(X_train.shape)
    print(delta.shape)