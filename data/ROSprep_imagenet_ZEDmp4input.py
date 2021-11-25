#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#


import jetson.utils

import argparse
import sys


cl_input = "data/ros_imagenet_furrows_ZEDmp4input.py --model=models/furrows_1f3f4f6f_batchsz32/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/furrows/labels.txt /home/vision-ii/Desktop/6furrow/lat_1.mp4"
# parse the command line
#parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
#                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() +
#                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.")


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





# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
font = jetson.utils.cudaFont()
    
    
import jetson.inference
# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)    

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


def inferencing():
    global count
    global left_count
    global mid_count 
    global right_count 
    global xleft_count 
    global xright_count 
    global left_20deg_count
    global left_40deg_count 
    global right_20deg_count 
    global right_40deg_count 
    global straight_count 

    # process frames until the user exits
    while True:
        	# capture the next image
        	img = input.Capture()
        
        	# classify the image
        	class_id, confidence = net.Classify(img)
        	
        
        	# find the object description
        	class_desc = net.GetClassDesc(class_id)
        	print("Image", count,confidence*100,"% - ",class_desc)
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
        	print("--------------------------------------------------")
        	print("% of Video Labeled - LEFT 20 DEG: ", left_20deg_count,"/",count,"=", left_20deg_count/count*100, "%")
        	print("% of Video Labeled - LEFT 40 DEG: ", left_40deg_count,"/",count,"=", left_40deg_count/count*100, "%")
        	print("% of Video Labeled - RIGHT 20 DEG: ", right_20deg_count,"/",count,"=", right_20deg_count/count*100, "%")
        	print("% of Video Labeled - RIGHT 40 DEG: ", right_40deg_count,"/",count,"=", right_40deg_count/count*100, "%")
        	print("% of Video Labeled - STRAIGHT: ", straight_count,"/",count,"=", straight_count/count*100, "%")
        	print("--------------------------------------------------")

        
        
        	# overlay the result on the image	
        	font.OverlayText(img, img.width, img.height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
        	
        	# render the image
        	output.Render(img)
        
        	# update the title bar
        	output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))
        
        	# print out performance info
        	net.PrintProfilerTimes()
        
        	# exit on input/output EOS
        	if not input.IsStreaming() or not output.IsStreaming():
        		break
        
if __name__ == "__main__":
    inferencing()


