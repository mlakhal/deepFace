'''
===================================================
Faces detection 
===================================================

Run command:
    face_detection.py -i img_path -w weight_path

author: @mlakhal

'''
import numpy as np
import cv2
import argparse

from models import FaceCNN

WIDTH = 32
HEIGHT = 32

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True,
                    help = 'Image path')
    ap.add_argument('-w', '--weights', required = True,
                    help = 'CNN weights path')
    args = vars(ap.parse_args())

    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    
    w_path = args['weights']
    model = FaceCNN(nb_class=7, lr=0.1, weights_path=w_path)

    img_path = args['image']
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    faces_dic = {0: 'Sharon', 1: 'Powell', 2: 'Rumsfeld',\
                3: 'W Bush', 4: 'Schroeder', 5: 'Chavez',\
                6: 'Blair'}
    # looping over the detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,204,0),2)
        # extract the face to be fed on our CNN
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (WIDTH, HEIGHT)) 
        X = face[np.newaxis, np.newaxis, :, :]
        pred_cls = model.predict_classes(X)
        #proba = model.predict_proba(X)
        
        xText = x - 10     
        ytext = y + h + 30
        label = faces_dic[pred_cls[0]]

        cv2.putText(img, label, (xText, ytext),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 204, 0), 2)

        #roi_img = img[y:y+h, x:x+w]

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Saving result
    s_path = img_path.split('/') # assume the path is of the forme "folder/img"
    filename = s_path[0] + '/res-' + s_path[1]
    cv2.imwrite(filename, img)

if __name__ == '__main__':
    main() 
