# Age_Gender_Race Prediction
Age, Gender and Race Prediction by Using GoogleNet (Inception) Architecture and extending the architecture to multi-output CNN 

**Steps for Running**<br/>
1 .First download and install all necessary libraries <br/>
   - [Python](https://www.python.org) 3.6.8 or any latest version <br/>
   - [opencv](https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html)<br/>
   - [Keras](https://pypi.org/project/Keras/) <br/>
   - [Tensorflow](https://www.tensorflow.org/install) <br/>
   - [dlib](https://pypi.org/project/dlib/) 
2. Next download the UTKFace dataset<br/>
   - [UTKFace](https://www.kaggle.com/jangedoo/utkface-new)<br/><br/>
3. Training <br/>
   - If you want to train the model from scratch use Colab [Notebook](https://github.com/BolluBalaji/Age_Gender_Race/blob/master/Mutli_CNN_Age_Gender_Race.ipynb)<br/>
   - or Use [Pretrained](https://drive.google.com/open?id=1FiHY3VPhbaRRsIEcft7EH4tAKh7m5-DG) Model<br/><br/>
 4. For Live Detection <br/>
   * i) [HaarCascade](https://github.com/BolluBalaji/Age_Gender_Race/blob/master/haarcascade_frontalface_default.xml) Face Detector<br/>
      - Run [HarCascade_Face_Recog.py](https://github.com/BolluBalaji/Age_Gender_Race/blob/master/HarCascade_Face_Recog.py)<br/>
  * ii)[HOG](https://pypi.org/project/hog/) Histogram of oriented gradients<br/>
      - Run [Hog_Face_Recog.py](https://github.com/BolluBalaji/Age_Gender_Race/blob/master/Hog_Face_Recog.py)<br/>

