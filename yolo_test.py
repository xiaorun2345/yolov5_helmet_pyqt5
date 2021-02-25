#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/20 18:19
# @Author  : xiaorun
# @Site    : 
# @File    : yoloDetect.py
# @Software: PyCharm
import sys
import threading
from threading import Thread
import time
import os
import cv2
from yolo import YOLO5
from flask import Flask, render_template, Response,request,redirect,url_for
import argparse
import json,jsonify
from gevent import monkey
monkey.patch_all()  # 打上猴子补丁
from gevent import pywsgi
camera = cv2.VideoCapture(0)
global objs
def parseData():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/exp0/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='output confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    return opt

def objDetector(frame):
    yolo = YOLO5()
    opt = parseData()
    yolo.set_config(opt.weights, opt.device, opt.img_size, opt.conf_thres, opt.iou_thres, True)
    yolo.load_model()
    objs=yolo.obj_detect(frame)
    for obj in objs:
        cls=obj["class"]
        cor=obj["color"]
        conf='%.2f' % obj["confidence"]
        label=cls+" "+str(conf)
        x,y,w,h=obj["x"],obj["y"],obj["w"],obj["h"]
        cv2.rectangle(frame, (int(x),int(y)), (int(x+w), int(y+h)), tuple(cor))
        cv2.putText(frame, label, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, thickness=2)
    person="there are {} person ".format(len(objs))
    cv2.putText(frame, person, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)
    #cv2.putText(frame, str(len(objs)), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=3)
    return frame

def gen_detector(flags):
    while True:
        ret,frame=camera.read()
        if flags==True:
            frame=objDetector(frame)
            cv2.imshow("test",frame)
            if cv2.waitKey(2)==ord("q"):
                break
if __name__ == '__main__':
    gen_detector(True)