import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


Trainfeatures=[]

for file in range(1,100):
    print(file)
    file_temp=file
    Img_1='Dataset\Img('
    Img_2=file_temp
    Img_ext=').png'
    Img1=Img_1+str(Img_2)+Img_ext
    
    import numpy as np
    imgg=cv2.imread(Img1)
    resize1=cv2.resize(imgg,(256,256))
        
    try:
        gray1=cv2.cvtColor(resize1,cv2.COLOR_BGR2GRAY)
    except:
        gray1=resize1

    M=np.mean(imgg)
    S=np.std(imgg)
    V=np.var(imgg)
    
    Features=[M,S,V]

    Trainfeatures.append(Features)


import pickle
with open('Trainfeatures.pickle','wb') as f:
    pickle.dump(Trainfeatures,f)
    
