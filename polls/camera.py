import os,urllib.request
from cv2 import cv2
import numpy as np
from django.conf import settings
from polls import object_detector
import threading


class VideoCamera(object):
    def __init__(self):
        # ipcamera rtsp protocol
        # self.video = cv2.VideoCapture('rtsp://admin:Abc1234*@192.168.1.108/cam/realmonitor?channel=1&subtype=1') # rtsp://admin:Abc1234*@192.168.1.108/cam/realmonitor?channel=1&subtype=1

        # ip camera
        # self.video = cv2.VideoCapture('http://admin:Abc1234*@192.168.1.108/cgi-bin/snapshot.cgi?channel=1')

        # local webcam
        self.video = cv2.VideoCapture(0)
        print('camera videCaptur(0) başlatildi')
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()
        print(threading.active_count())
        self.detect = object_detector.ObjectDetection()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        image_detected = self.detect.object_detection(image)
        # cv2.imshow('resim', image_detected)
        # cv2.waitKey(0)
        ret, jpeg = cv2.imencode('.jpg', image_detected)
        # print('resim alindi jpeg e çevirildi')
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')