import cv2
import face_recognition
from PIL import Image
import numpy as np
import dlib
from scipy.io.wavfile import write
import sys
from sonipy.sonify import SonifyTool
import pyOSC3
import time, random
import keras
from keras.models import load_model
import sys
import tensorflow.compat.v1 as tf
capture = cv2.VideoCapture(0)
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
factor = 4
client = pyOSC3.OSCClient()
client.connect( ( 'localhost', 3333 ) )
isClosed = False

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2


from model import predict, image_to_tensor, deepnn
modelPath = './ckpt'
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
tf.disable_eager_execution()
face_x = tf.placeholder(tf.float32, [None, 2304])
y_conv = deepnn(face_x)
probs = tf.nn.softmax(y_conv)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(modelPath)
sess = tf.Session()
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')



def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def detect_face_bb(image, detector):
    small_frame = cv2.resize(image, (0, 0), fx=1/factor, fy=1/factor)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    return rects

def detect_face_blob(image):
    small_frame = cv2.resize(image, (0, 0), fx=1/factor, fy=1/factor)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    face_landmarks_list = face_recognition.face_landmarks(gray)
    return face_landmarks_list

def detect_face_bb_blob(image, detector):
    small_frame = cv2.resize(image, (0, 0), fx=1/factor, fy=1/factor)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    face_landmarks_list = face_recognition.face_landmarks(gray)
    return rects, face_landmarks_list

def plot_res_bb(image, rects):
    res_list = []
    if len(rects) == 0:
        image1 = image
    else:
        for faceRect in rects:
            x1 = faceRect.rect.left()
            y1 = faceRect.rect.top()
            x2 = faceRect.rect.right()
            y2 = faceRect.rect.bottom()
            image1 = cv2.rectangle(image, (factor*x1-10,factor*y1), (factor*x2+10, factor*y2+20), (0, 0,255), 2)
            res_list.append([x1-5,y1-5,x2+5,y2+5])
            #image1 = cv2.putText(image1,'face3',(x1 - 10, y1 - 10),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2,1)
    return image1,res_list

def plot_res_blob(image, face_landmarks_list):
    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            a = []
            for i in face_landmarks[facial_feature]:
                a.append([factor*i[0],factor*i[1]])
            a = np.array(a,np.int32)
            a = a.reshape((-1, 1, 2))
            image = cv2.polylines(image, [a], isClosed, color, thickness)
    return image

def crop_image(org, faceRect):
    x1 = faceRect.rect.left()
    y1 = faceRect.rect.top()
    x2 = faceRect.rect.right()
    y2 = faceRect.rect.bottom()
    image1 = org[factor*y1: factor*y2+20,factor*x1-10:factor*x2+10]
    return image1

def compute_pixel_list(img):
    pixel_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_list.append(img[i,j])
    return pixel_list
def filter(f):
    f_new = []
    for i in f:
        if abs(i)>1000000:
            i=0
        if abs(i)<1000:
            i=0
        f_new.append(i)
    return f_new
def generate_sound_wave(pixel_list):
    sound_wave = np.real(2*(pixel_list-min(pixel_list))/(max(pixel_list)-min(pixel_list))-1)
    #scaled = np.int16(sound_wave/np.max(np.abs(sound_wave)) * 32767)

    return sound_wave.tolist()
def predict_emotion(face_image, model):
    face_image = cv2.resize(face_image, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v,k) for k,v in emotion_dict.items())
    predicted_label = label_map[predicted_class]
    return predicted_label
def emotion_detect(detected_face,probs, showBox=False):

    tensor = image_to_tensor(detected_face)
    result = sess.run(probs, feed_dict={face_x: tensor})

    return result


frame_count = 0
scale_all = []
while True:
    frame_count += 1
    ret, frame = capture.read()
    rects, face_landmarks_list = detect_face_bb_blob(frame,detector)
    #face_landmarks_list = detect_face_blob(frame)
    res, res_list = plot_res_bb(frame, rects)
    res = plot_res_blob(res,face_landmarks_list)
    cv2.imshow('frame', res)
    if frame_count%3==0:
        msg = pyOSC3.OSCMessage()
        msg.setAddress("/"+"frameCount")
        msg.append(frame_count)
        client.send(msg)
    if len(rects) > 0:
        count = 0
        for faceRect in rects:
            res_crop = crop_image(frame, faceRect)
            try:
                img = cv2.resize(res_crop,[48,48])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                emotions = emotion_detect(img,probs)[0].tolist()
                print(emotions)
                if frame_count%3==0:
                    for i in range(len(emotions)):
                        msg = pyOSC3.OSCMessage()
                        msg.setAddress("/"+"emotion_"+EMOTIONS[i])
                        msg.append(emotions[i])
                        client.send(msg)
                res = cv2.resize(res_crop,[30,30])
                B = res[:,:,0]
                pixel_list_B = compute_pixel_list(B)
                '''
                f_B = np.fft.fft(pixel_list_B)
                f_new_B = filter(f_B)
                pixel_new_B = np.fft.ifft(f_new_B)
                '''
                scaled = generate_sound_wave(pixel_list_B)
                if len(face_landmarks_list)>0:
                    x1 = res_list[count][0]
                    y1 = res_list[count][1]
                    x2 = res_list[count][2]
                    y2 = res_list[count][3]
                    for facial_feature in face_landmarks_list[count].keys():
                        res_facial_list_y = []
                        res_facial_list_x = []
                        for i in face_landmarks_list[count][facial_feature]:

                            res_facial_list_y.append((i[1]-y1)/(y2-y1))
                            res_facial_list_x.append((i[0]-x1)/(x2-x1))
                        # set blob from osc (random type),or do all the audio in python
                        # resterization: change is suttle
                        # FACE PIXEL + EMOTION TO DECIDE DIFFERENT FILTERS

                        msg = pyOSC3.OSCMessage()
                        msg.setAddress("/"+facial_feature+"_y")
                        msg.append(res_facial_list_y)
                        client.send(msg)
                        msg = pyOSC3.OSCMessage()
                        msg.setAddress("/"+facial_feature+"_x")
                        msg.append(res_facial_list_x)
                        client.send(msg)
            except:
                continue

            count = count+1

            msg = pyOSC3.OSCMessage()
            msg.setAddress("/print")
            msg.append(scaled)
            client.send(msg)

    if cv2.waitKey(1) == ord('q'):
        break
