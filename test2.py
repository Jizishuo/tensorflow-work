# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:23:59 2018

@author: User
"""
        
from multiprocess import Process
import os
import time
import numpy as np

def hobby_motion(name):
    print('%s喜欢运动'% name)
    print ('Child process with processId %s starts.' % os.getpid())
    time.sleep(np.random.randint(1,3))

def hobby_game(name):
    print('%s喜欢游戏'% name)
    print ('Child process with processId %s starts.' % os.getpid())
    time.sleep(np.random.randint(1,3))

if __name__ == "__main__":
    print ('Parent processId is: %s.' % os.getpid())

    p1 = Process(target=hobby_motion, args=('付婷婷',))
    p2 = Process(target=hobby_game, args=('kebi',))
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    