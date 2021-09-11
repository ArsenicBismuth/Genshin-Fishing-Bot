import cv2
import numpy as np
import mouse
import time
from mss import mss
from threading import Thread, active_count

# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
# https://stackoverflow.com/questions/34712480/finding-an-icon-in-an-image
# https://stackoverflow.com/questions/8533318/multiprocessing-pool
# https://stackoverflow.com/questions/58293187/opencv-real-time

off = [0,-45*0]  # Set 0 to 1 if not in event fishing spot
co = [740,1175,220+off[1],255+off[1]] # x1,x2,y1,y2
mon = {"top": co[2], "left": co[0], "width": co[1]-co[0], "height": co[3]-co[2]}
sct = mss()     # Screenshot
fps_limit = 60  # fps limit

# Async take screenshot
class ThreadedScreen(object):
    def __init__(self):

        self.capture = self.image()
        
        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()
        
        self.frame = None

    def update(self):
        while True:
            start = time.time()
            self.frame = self.image()
            # print(1.0/(time.time() - start)) # FPS

    def grab_frame(self):
        return self.frame

    def image(self):
        # Screenshot then preprocess
        img = np.asarray(sct.grab(mon)) # Capture smaller region, 10 FPS => 60 FPS
        # img = np.asarray(sct.grab(sct.monitors[0])) # Capture all region, too slow
        
        # img = img[co[2]:co[3], co[0]:co[1], :] # Crop, y1:y2, x1:x2
        img = img[:, :, :3]   # Remove alpha
        # img = img[::2, ::2, :] # Downscale
        # img = img[:, :, ::-1] # Reverse BGR => RGB
        return img

# Async template matching
class ThreadedLocate(object):
    def __init__(self, img, t, m):
    
        self.t = t
        self.m = m
        self.img = img
        
        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()
        
        self.loc = None
        self.value = None

    def update(self):
        while True:
            self.loc, self.value = self.locate(self.img, self.t, self.m)
            # print(1.0/(time.time() - start)) # FPS
    
    def set(self, img):
        self.img = img
        # Don't put update() here (a.k.a update on change).
        # because that's pratically pooling: change => wait update.

    def get(self):
        return self.loc, self.value

    def locate(self, img, template, mask):
        # img = preproc(image())
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Return center point from closest match
        w, h = template.shape[::-2]
        return np.array([max_loc[0] + w/2, max_loc[1] + h/2]), max_val
    
    
    
def preproc(img):
    # img = img[:, :, 0] # Extract blue only (more contrast vs grayscale)
    return img
    
def printbar(bar, left, right):
    k = 1/20
    d = {"|":int(bar*k), "<":int(left*k), ">":int(right*k)}
    s = sorted(d.items(), key=lambda x: x[1])
    
    print("  [  " + " "*
        s[0][1]             + s[0][0] + " "*
        (s[1][1]-s[0][1])   + s[1][0] + " "*
        (s[2][1]-s[1][1])   + s[2][0] + " "*
        (int((co[1]-co[0])*k)-s[2][1])+ "  ]", end=" ")
    
def main():
    # Load templates
    bar_t = preproc(cv2.imread("./template/bar.png"))
    left_t = preproc(cv2.imread("./template/left.png"))
    right_t = preproc(cv2.imread("./template/right.png"))
    # Load masks
    bar_m = preproc(cv2.imread("./template/bar_mask.png"))
    left_m = preproc(cv2.imread("./template/left_mask.png"))
    right_m = preproc(cv2.imread("./template/right_mask.png"))
    
    # Initialize threads
    capture = ThreadedScreen()
    # mouse = ThreadedClick()
    time.sleep(1) # Wait till thread get result
    
    img = capture.grab_frame()
    
    bar_th = ThreadedLocate(img, bar_t, bar_m)
    left_th = ThreadedLocate(img, left_t, left_m)
    right_th = ThreadedLocate(img, right_t, right_m)
    bar_th.set(img)
    left_th.set(img)
    right_th.set(img)
    time.sleep(1) # Wait till thread get result
    
    while True:
        start = time.time()
        
        # Grab image from thread (10 => 150 FPS)
        img = capture.grab_frame()        
        bar_th.set(img)
        left_th.set(img)
        right_th.set(img)
        
        # Template matching (threading: 10 => 30 FPS)
        bar, bar_val = bar_th.get()
        left, left_val = left_th.get()
        right, right_val = right_th.get()
        
        # Target is between the two limiters
        target = left + (right-left) * 0.5
        val = min(bar_val, left_val, right_val)
        
        # Validations
        valid = True
        if (right[0] < left[0]):
            valid = False
        if (val < 0.4):
            valid = False
        
        # Mouse
        if (valid):
            
            # x as set point
            if (bar[0] <= target[0] * 1.0):
                mouse.press("left")
                # pyautogui.mouseDown(co[0],co[2])
                
                # Hold, don't stop till overshoot
            elif (bar[0] > target[0] * 1.0):
                mouse.release("left")
                # pyautogui.mouseUp()
                
            # Utilities
            printbar(bar[0], left[0], right[0])
            # print(" %.1fFPS"%fps, end=" ")
            print()
        
        # Utilities
        frame_time = time.time() - start
        if fps_limit>0:
            time.sleep(max(1.0/fps_limit - frame_time, 0))
        fps = 1.0 / (time.time() - start)
                
        # print(active_count(), end="")
        # print(" %.1fFPS"%fps, end=" ")
        # print("%.1fFPS"%fps, "%.2f"%val, target[0], bar[0], right[0])
        # print("%.2f"%val, target[0], bar[0], right[0])
        # print()
        
if __name__ == "__main__":
    main()