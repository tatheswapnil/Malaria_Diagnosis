# =========================== IMPORT PACKAGES =================================

import matplotlib.pyplot as plt
import tkinter
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')
import streamlit as st


import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('imgg.jpeg')


st.title('Malaria Disease Cell Classification ')

genre = st.radio(
    "You didn't select Any Data.",
    ('NO', 'YES'))

if genre == 'NO':
    st.write("Select Yes to Import Data")
else:
    file_up = st.file_uploader("Upload An Image",type = "png")
    
    if file_up==None:
        st.text("BROWSE IMAGE")
    
    else:
    
        # ============================ INPUT IMAGE ====================================
    
        imgg=mpimg.imread(file_up)
        plt.imshow(imgg)
        plt.axis('off')
        plt.title("Original image")
        plt.show()
        st.image(imgg, caption='Input Image')
    
        # ============================ PREPROCESSING ==================================
    
        # ==== Image resize ====
    
        import numpy as np
        resize1=cv2.resize(imgg,(256,256))
        plt.imshow(resize1)
        plt.axis('off')
        plt.title("Resized image")
        plt.show()
        st.image(resize1, caption='Resized Image')
    
    
        # ==== Image Grayscale ====
    
        import cv2
        try:
            gray1=cv2.cvtColor(resize1,cv2.COLOR_BGR2GRAY)
        except:
            gray1=resize1
                                               
        plt.imshow(gray1)
        plt.axis('off')
        plt.title("Grayscale image")
        plt.show()
        st.image(gray1, caption='Grayscale Image')
    
        #plt.show()
        # ====================== FEATURE EXTRACTION ==================================
    
        M=np.mean(imgg)
        S=np.std(imgg)
        V=np.var(imgg)
        Features=[M,S,V]

        
        
#        print("==============================================================")
#        print("        Feature Extraction --> Mean Standard Variance         ")
#        print("==============================================================")
#        print()
#        print(" 1. Mean     =",M)
#        print()
#        print(" 2. Standard =",S)
#        print()
#        print(" 3. Variance =",V)
#    
#    
#        st.text("==============================================================")
#        st.text("        Feature Extraction --> Mean Standard Variance         ")
#        st.text("==============================================================")
#        st.write(" 1. Mean     =",M)
#        st.write(" 2. Standard =",S)
#        st.write(" 3. Variance =",V)
    
    
    
    
        # ===== LOAD A PICKLE FILE ======
    
        import pickle
        with open('Trainfeatures.pickle','rb') as fp:
            Trainfeatures=pickle.load(fp)
            
        y_trains=np.arange(0,99)
        y_trains[0:50]=0
        y_trains[50:99]=1

    
    
        # ================================ CLASSIFICATION ============================
    
        # ==== BAGGING CLASSIFIER =====
    
        from sklearn import metrics
    
        from sklearn.ensemble import BaggingClassifier
    
        #initialize the model 
        bagg = BaggingClassifier()
    
        #fitting the model
        bagg.fit(Trainfeatures, y_trains)  
    
        #predict the model
        pred_bagg = bagg.predict([Features])
    
        prediction1_bagg = bagg.predict(Trainfeatures)
    
    
    
        print("==============================================================")
        print("                         Prediction                           ")
        print("==============================================================")
        print()
    
        if pred_bagg==0:
            print("--------------------------------------------------")
            print("      Cell Identifed  = AFFECTED - (Parasitized)")
            print("--------------------------------------------------")
    
        elif pred_bagg==1:
            print("-----------------------------------")
            print("   Cell Identifed   = NOT AFFCETED"    )
            print("-----------------------------------")
    
            

        
        st.text("=============================================================")
#        st.text("==============================================================")
#        st.text("                         Prediction                           ")
#        st.text("==============================================================")
    
        if pred_bagg==0:
            print("-----------------------------------")
            print("     The Identifed  = AFFECTED - (Parasitized)   ")
            print("-----------------------------------")
            predd="     The Identifed  = AFFECTED - (Parasitized)    "
    
        elif pred_bagg==1:
            print("-----------------------------------")
            print("The Identifed   = NOT AFFCETED"    )
            print("-----------------------------------")
            predd=" The Identifed   = NOT AFFCETED"
    
            
    
        variable_output = str(predd)
        
        font_size = 18
        
        html_str = f"""
        <style>
        p.a {{
          font: bold {font_size}px Courier;
        }}
        </style>
        <p class="a">{variable_output}</p>
        """
        
        st.markdown(html_str, unsafe_allow_html=True)





        #======================== PERFORMANCE METRICS ==============================
    
        print("==============================================================")
        print("                   Performance Estimations                    ")
        print("==============================================================")
        print()
    
    
        acc1_bagg=metrics.accuracy_score(prediction1_bagg,y_trains)*100
    
        print("1.Accuracy    =", acc1_bagg,'%')
        print()
        error=100-acc1_bagg
        print("2.Error Rate  =", error)
        print()
        print("3.Classification Metrics")
        print()
        print(metrics.classification_report(prediction1_bagg,y_trains))
    
        print()
        print("----------------------------------------------------------------------")
        print()
    

        
        
        st.text("=============================================================")
    
#        st.text("==============================================================")
#        st.text("                   Performance Estimations                    ")
#        st.text("==============================================================")
    
        acc1_bagg=metrics.accuracy_score(prediction1_bagg,y_trains)*100
   
        variable_output = '1. ACCURACY = '+str(acc1_bagg)
        
        font_size = 18
        
        html_str = f"""
        <style>
        p.a {{
          font: bold {font_size}px Courier;
        }}
        </style>
        <p class="a">{variable_output}</p>
        """
        
        st.markdown(html_str, unsafe_allow_html=True)
        
        
    
#        st.write("1.Accuracy    =", acc1_bagg,' %')
        
        error=100-acc1_bagg
        
        variable_output = '2. ERROR RATE = '+str(error)
        
        font_size = 18
        
        html_str = f"""
        <style>
        p.a {{
          font: bold {font_size}px Courier;
        }}
        </style>
        <p class="a">{variable_output}</p>
        """
        
        st.markdown(html_str, unsafe_allow_html=True)        
        
        
        
#        st.write("2.Error Rate  =", error,' %')
#        st.write("3.Classification Metrics")
#        st.write(metrics.classification_report(prediction1_bagg,y_trains))
    
#        agree = st.checkbox('Plot Performance Graph')
    
#        if agree:

        import matplotlib.pyplot as plt
        data={'Accuracy':acc1_bagg,'Error Rate':error}
        value1=list(data.keys())
        value2=list(data.values())
        fig=plt.figure(figsize=(5,4))
        plt.bar(value1,value2,color='maroon',width=0.35)
        plt.xlabel("Performance Estimations")
        plt.title("Bagging Classifier")
        plt.grid()
        plt.savefig("graph.png")
        plt.show()
        
        st.image("graph.png")

