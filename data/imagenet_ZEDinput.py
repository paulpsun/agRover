import jetson.inference
import jetson.utils
from jetson.utils import cudaFromNumpy as cfn, saveImageRGBA
import argparse
import sys
import base64
import numpy as np

import cv2
import pyzed.sl as sl
import os

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)



##########################################
def main():

    #Global Variable Counters
    count = 0
    left_count = 0
    mid_count = 0
    right_count = 0
    xleft_count = 0
    xright_count = 0
    left_20deg_count = 0
    left_40deg_count = 0
    right_20deg_count = 0
    right_40deg_count = 0
    straight_count = 0
    ############ZED###############
    
    print("Running...")
    init = sl.InitParameters()
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    print_camera_information(cam)
    print_help()  
    
    ####RECORD WITH CV2###########
    filename = input("Enter file name: ")
    image_size = cam.get_camera_information().camera_resolution
    width = int(image_size.width )
    height = int(image_size.height )
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (width,height))
    ##############################
	# load the recognition network
    net = jetson.inference.imageNet(opt.network, sys.argv)
    ##############################
 
    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
        
            cam.retrieve_image(mat, sl.VIEW.LEFT)
			################################################
            image_data = mat.get_data()
            
            cuda_mem = cfn(image_data)
            # classify the image
            class_id, confidence = net.Classify(cuda_mem)

            # find the object description
            class_desc = net.GetClassDesc(class_id)

            
            print("Image", count,confidence*100,"% -------------------------------",class_desc)
       
            count+=1
            
            if class_desc == "left":
                left_count+=1
            if class_desc == "mid":
                mid_count+=1
            if class_desc == "right":
                right_count+=1
            if class_desc == "xleft":
                xleft_count+=1
            if class_desc == "xright":
                xright_count+=1
            #---------------------------
            if class_desc == "left_20deg":
                left_20deg_count+=1
            if class_desc == "left_40deg":
                left_40deg_count+=1
            if class_desc == "right_20deg":
                right_20deg_count+=1
            if class_desc == "right_40deg":
                right_40deg_count+=1
            if class_desc == "straight":
                straight_count+=1
            #---------------------------
            
            print("% of Video Labeled - LEFT: ", left_count,"/",count,"=", left_count/count*100, "%")
            print("% of Video Labeled - MID: ", mid_count,"/",count,"=", mid_count/count*100, "%")
            print("% of Video Labeled - RIGHT: ", right_count,"/",count,"=", right_count/count*100, "%")
            print("% of Video Labeled - XLEFT: ", xleft_count,"/",count,"=", xleft_count/count*100, "%")
            print("% of Video Labeled - XRIGHT: ", xright_count,"/",count,"=", xright_count/count*100, "%")
            print("--------------------")
            print("% of Video Labeled - LEFT 20 DEG: ", left_20deg_count,"/",count,"=", left_20deg_count/count*100, "%")
            print("% of Video Labeled - LEFT 40 DEG: ", left_40deg_count,"/",count,"=", left_40deg_count/count*100, "%")
            print("% of Video Labeled - RIGHT 20 DEG: ", right_20deg_count,"/",count,"=", right_20deg_count/count*100, "%")
            print("% of Video Labeled - RIGHT 40 DEG: ", right_40deg_count,"/",count,"=", right_40deg_count/count*100, "%")
            print("% of Video Labeled - STRAIGHT: ", straight_count,"/",count,"=", straight_count/count*100, "%")
                       
            # print out performance info
            # net.PrintProfilerTimes()
            print("==============================================================================")
            print("==============================================================================")
            
            ################################################
            ####OVERLAY FONT################################
            
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            topLeftCornerOfText = (10,30)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2

            cv2.putText(image_data,"{:05.2f}% {:s}".format(confidence * 100, class_desc), 
                topLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            
            ################################################            
            #Delete the 4th element of the 3rd axis, it contains the depth information pulled by ZED
            image_data = np.delete(image_data,3,2)
            
            #out.write(image_data)
            cv2.imshow("ZED", image_data)
            
            key = cv2.waitKey(5)
            settings(key, cam, runtime, mat)
        else:
            key = cv2.waitKey(5)
            break
            
    cam.close()   
    out.release()
    cv2.destroyAllWindows()
    print("\nFINISH")


def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
    print("Camera FPS: {0}.".format(cam.get_camera_information().camera_fps))
    print("Firmware: {0}.".format(cam.get_camera_information().camera_firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Record a video:                     z")
    print("  Quit:                               q\n")


def settings(key, cam, runtime, mat):
    if key == 115:  # for 's' key
        switch_camera_settings()
    elif key == 43:  # for '+' key
        current_value = cam.get_camera_settings(camera_settings)
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cam.get_camera_settings(camera_settings)
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
        print("Camera settings: reset")
    elif key == 122:  # for 'z' key
        record(cam, runtime, mat)


def switch_camera_settings():
    global camera_settings
    global str_camera_settings
    if camera_settings == sl.VIDEO_SETTINGS.BRIGHTNESS:
        camera_settings = sl.VIDEO_SETTINGS.CONTRAST
        str_camera_settings = "Contrast"
        print("Camera settings: CONTRAST")
    elif camera_settings == sl.VIDEO_SETTINGS.CONTRAST:
        camera_settings = sl.VIDEO_SETTINGS.HUE
        str_camera_settings = "Hue"
        print("Camera settings: HUE")
    elif camera_settings == sl.VIDEO_SETTINGS.HUE:
        camera_settings = sl.VIDEO_SETTINGS.SATURATION
        str_camera_settings = "Saturation"
        print("Camera settings: SATURATION")
    elif camera_settings == sl.VIDEO_SETTINGS.SATURATION:
        camera_settings = sl.VIDEO_SETTINGS.SHARPNESS
        str_camera_settings = "Sharpness"
        print("Camera settings: Sharpness")
    elif camera_settings == sl.VIDEO_SETTINGS.SHARPNESS:
        camera_settings = sl.VIDEO_SETTINGS.GAIN
        str_camera_settings = "Gain"
        print("Camera settings: GAIN")
    elif camera_settings == sl.VIDEO_SETTINGS.GAIN:
        camera_settings = sl.VIDEO_SETTINGS.EXPOSURE
        str_camera_settings = "Exposure"
        print("Camera settings: EXPOSURE")
    elif camera_settings == sl.VIDEO_SETTINGS.EXPOSURE:
        camera_settings = sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE
        str_camera_settings = "White Balance"
        print("Camera settings: WHITEBALANCE")
    elif camera_settings == sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE:
        camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
        str_camera_settings = "Brightness"
        print("Camera settings: BRIGHTNESS")


def record(cam, runtime, mat):
    vid = sl.ERROR_CODE.FAILURE
    out = False
    while vid != sl.ERROR_CODE.SUCCESS and not out:
        filepath = input("Enter filepath name: ")
        record_param = sl.RecordingParameters(filepath)
        vid = cam.enable_recording(record_param)
        print(repr(vid))
        if vid == sl.ERROR_CODE.SUCCESS:
            print("Recording started...")
            out = True
            print("Hit spacebar to stop recording: ")
            key = False
            while key != 32:  # for spacebar
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat)
                    cv2.imshow("ZED", mat.get_data())
                    key = cv2.waitKey(5)
        else:
            print("Help: you must enter the filepath + filename + SVO extension.")
            print("Recording not started.")
    cam.disable_recording()
    print("Recording finished.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
