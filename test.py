import numpy as np
import pyautogui
import time
import cv2
from multiprocessing import Pool, freeze_support
from mss import mss
from threading import Thread

resolutions = [
    (0, 0, 100,100),(0, 0, 200,100),
    (0, 0, 200,200),(0, 0, 400,200),
    (0, 0, 400,400),(0, 0, 800,400),
    (0, 0, 1080,4100),
    (220,740,1175,255)
]
# shape = resolutions[0]
shape = resolutions[7]

def main():    
    sct = mss()
    
    # Turns out sct screenshot can be smaller region
    # and much faster, 10-20 FPS to astounding 60 FPS
    # mon = sct.monitors[0]
    # print(mon)
    
    while True:
        start = time.time()
        mon = {"top": shape[0], "left": shape[1], "width": shape[2]-shape[1], "height": shape[3]-shape[0]}
        frame = np.asarray(sct.grab(mon))
        
        cv2.imshow("", frame)
        cv2.waitKey(0) 
        
        print(1.0/(time.time() - start))

if __name__ == '__main__':
    main()