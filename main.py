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
# https://stackoverflow.com/questions/52735231/how-to-select-all-non-black-pixels

off = 1  # Set 0 for event fishing
co = [740,1175,220-45*off,255] # x1,x2,y1,y2
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
            limitfps(start, fps_limit) # 40 FPS limit => 30 FPS
            # printfps(start)

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
        img = preproc(img)
        
        # cv2.imshow("", img)
        # cv2.waitKey(0) 
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
            start = time.time()
            self.loc, self.value = self.locate(self.img, self.t, self.m)
            limitfps(start, fps_limit)  # 500 FPS unlocked
            # printfps(start)
    
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
    
    
def printfps(ptime, end="\n"):
    d = time.time() - ptime
    fps = 0 if (d == 0) else 1.0/d
    print("%.1f"%fps, end=end) # FPS

def limitfps(ptime, limit):
    d = time.time() - ptime # frame_time
    if limit>0:
        time.sleep(max(1.0/limit - d, 0))
    
def display(img, bar, left, right, target):

    img = np.ascontiguousarray(img, dtype=np.uint8)

    # Point location display (x,y)
    cv2.circle(img, (int(bar[0]), int(bar[1])), radius=0, color=(0, 0, 255), thickness=4)
    cv2.circle(img, (int(left[0]), int(left[1])), radius=0, color=(0, 255, 0), thickness=4)
    cv2.circle(img, (int(right[0]), int(right[1])), radius=0, color=(255, 0, 0), thickness=4)
    
    cv2.imshow('Fishing', img)

def preproc(img):
    # Extract blue only (more contrast vs grayscale)
    # img = img[:, :, 0] 
    # Remove anything not yellow by masking using boolean array.
    img[~np.all(img == [192,255,255], axis=-1)] = [0,0,0]
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
    bar_m = cv2.imread("./template/bar_mask.png")
    left_m = cv2.imread("./template/left_mask.png")
    right_m = cv2.imread("./template/right_mask.png")
    
    # Initialize threads
    print("Initializing threads...")
    capture = ThreadedScreen()
    # mouse = ThreadedClick()
    time.sleep(1) # Wait for initial result
    
    img = capture.grab_frame()
    
    bar_th = ThreadedLocate(img, bar_t, bar_m)
    left_th = ThreadedLocate(img, left_t, left_m)
    right_th = ThreadedLocate(img, right_t, right_m)
    time.sleep(1) # Wait for initial result
    
    # Indexing
    n = 0
    
    print("Started.")
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
        width = right-left
        target = left + width * 0.5
        val = min(bar_val, left_val, right_val)
        
        # Validations
        valid = True
        if (right[0] < left[0]):
            valid = False
        if (val < 0.4):
            valid = False
        
        # Mouse
        if (valid):
            
            # Utilities
            printbar(bar[0], left[0], right[0])
            # printfps(start, " ")
            
            # Constant control system
            if (0 < target[0]-bar[0] < width[0]*0.5):
                if (n == 0):
                    print("o", end="")
                    mouse.click("left")
                elif (n >= 2):
                    n = -1
                n += 1
            
            # x as set point
            elif (bar[0] <= target[0] * 1.0):
                print("oo", end="")
                mouse.press("left")
                # pyautogui.mouseDown(co[0],co[2])
                
            # Hold, don't stop till overshoot
            elif (bar[0] > target[0] * 1.0):
                print("", end="")
                mouse.release("left")
                # pyautogui.mouseUp()
                
            print()
        
        # Utilities
        display(img, bar, left, right, target)
        
        # Wait 1ms to accept input before continuing
        key = cv2.waitKey(1)
        # print(key)
        if key == 27: # ESC
            break
            
        limitfps(start, fps_limit)
        # printfps(start, " ")
        # print()
        
if __name__ == "__main__":
    main()