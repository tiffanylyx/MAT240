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

client2 = pyOSC3.OSCClient()
client2.connect( ( 'localhost', 3000 ) )

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

def send_message(address, value):
    msg = pyOSC3.OSCMessage()
    msg.setAddress("/"+address)
    msg.append(value)
    client.send(msg)

    return

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
    width = frame.shape[1]/factor
    height = frame.shape[0]/factor
    '''
    if frame_count%3==0:
        msg = pyOSC3.OSCMessage()
        msg.setAddress("/"+"frameCount")
        msg.append(frame_count)
        #client.send(msg)
    '''
    if len(rects) > 0:
        count = 0
        for faceRect in rects:

            res_crop = crop_image(frame, faceRect)
            try:
            #if 1>0:
                size = faceRect.rect.right()-faceRect.rect.left()
                send_message('face_size', size/width)
                img = cv2.resize(res_crop,[48,48])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                emotions = emotion_detect(img,probs)[0].tolist()
                for i in range(len(emotions)):
                    send_message("emotion_"+EMOTIONS[i],emotions[i])

                res = cv2.resize(res_crop,[50,50])
                B = res[:,:,0]
                pixel_list_B = compute_pixel_list(B)
                b = np.asarray(pixel_list_B).astype("uint8").tobytes()


                msg = pyOSC3.OSCBundle()
                msg.setAddress("OSCBlob")
                msg.append(b,typehint = 'b')
                client2.send(msg)

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

                    face_center_x = (x1+x2)/(2*width)
                    face_center_y = (y1+y2)/(2*height)
                    send_message('face_center_x', face_center_x)
                    send_message('face_center_y', face_center_y)

                    # Compute Lip Size
                    top_lip = face_landmarks_list[count]['top_lip']
                    min_y = min(i[1] for i in top_lip)
                    min_x =  min(i[0] for i in top_lip)
                    max_x =  max(i[0] for i in top_lip)
                    bottom_lip = face_landmarks_list[count]['bottom_lip']
                    max_y = max(i[1] for i in bottom_lip)
                    lip_size = (max_y-min_y)/(max_x-min_x)
                    send_message('lip_size', lip_size)



                    # Compute Left Eye Size
                    left_eye = face_landmarks_list[count]['left_eye']
                    min_y = min(i[1] for i in left_eye)
                    max_y = max(i[1] for i in left_eye)
                    min_x = min(i[0] for i in left_eye)
                    max_x = max(i[0] for i in left_eye)
                    left_eye_size = (max_y-min_y)/(max_x-min_x)
                    send_message('left_eye_size', left_eye_size)


                    # Compute Right Eye Size
                    right_eye = face_landmarks_list[count]['right_eye']
                    min_y = min(i[1] for i in right_eye)
                    max_y = max(i[1] for i in right_eye)
                    min_x = min(i[0] for i in right_eye)
                    max_x = max(i[0] for i in right_eye)
                    right_eye_size = (max_y-min_y)/(max_x-min_x)
                    send_message('right_eye_size', right_eye_size)


                    # Compute noise angle
                    nose_bridge = face_landmarks_list[count]['nose_bridge']
                    first = nose_bridge[0]
                    last = nose_bridge[-1]
                    nose_bridge_k = (last[1]-first[1])/max(1,(last[0]-first[0]))
                    send_message('nose_bridge_k', abs(nose_bridge_k))




            except:
                print('here')




            count = count+1

            msg = pyOSC3.OSCMessage()
            msg.setAddress("/print")
            msg.append(11)
            client.send(msg)


    if cv2.waitKey(1) == ord('q'):
        break
