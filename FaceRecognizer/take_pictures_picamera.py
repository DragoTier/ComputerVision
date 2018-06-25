import cv2
import picamera
from picamera.array import PiRGBArray 
import time

print("Starting camera...")

camera = picamera.PiCamera()
raw_capture = PiRGBArray(camera)

for i in range(0,50):
	
	time.sleep(1)
	camera.capture(str(i)+"dennis.jpg")
	print(i)
