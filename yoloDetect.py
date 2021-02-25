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
app = Flask(__name__,template_folder='./templates')
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
    print(objs)
    for obj in objs:
        cls=obj["class"]
        cor=obj["color"]
        conf='%.2f' % obj["confidence"]
        label=cls+" "+str(conf)
        x,y,w,h=obj["x"],obj["y"],obj["w"],obj["h"]
        cv2.rectangle(frame, (int(x),int(y)), (int(x+w), int(y+h)), tuple(cor))
        cv2.putText(frame, label, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, thickness=2)

    return frame

def gen_detector(flags):
    while True:
        ret,frame=camera.read()
        if flags==True:
            frame=objDetector(frame)
        _, frame = cv2.imencode('.JPEG',frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame.tostring() + b'\r\n')



@app.route('/camera_detector')
def video_feed():
    return Response(gen_detector(True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_raw')
def video_raw():
    return Response(gen_detector(False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/config')
def parseConfig():
    if request.method == 'GET':  # 请求方式是get
        iou_thresh= request.args.get('iou_thresh')  # args取get方式参数
        conf= request.args.get('conf')
        source = request.args.get('source')

        return iou_thresh,conf,source

@app.route('/')
def index():  # 视图函数
    return render_template('login.html')




@app.route('/person')  # 代表个人中心页
def login():  # 视图函数
    if request.method == 'GET':  # 请求方式是get
        name = request.args.get('username')  # args取get方式参数
        password = request.args.get('password')
        if (name=="admin" and password=="admin"):
            return render_template('index.html')
            #return redirect(url_for('index.html'))
        else:
            return render_template('login.html')



@app.route('/data')
def data_response():
    global objs
    data=jsonify(objs)
    return data

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 7000), app)
    server.serve_forever()
    #app.run(host='127.0.0.1', port=7000)