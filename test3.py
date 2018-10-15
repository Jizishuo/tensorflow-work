# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:26:44 2018

@author: User
"""

from multiprocessing import Process, Pool
import time
import os

def func(i):
    print ("Child process start, %s" % time.ctime())
    time.sleep(i)
    print("chiild process id is :", os.getpid())
    print ("Child process end, %s" % time.ctime())


if __name__ == "__main__":
    print ("Parent process start, %s" % time.ctime())
    print ("Parent process id",os.getpid() )
    start = time.time()
    pool = Pool(8)                # 创建进程池对象，进程数与multiprocessing.cpu_count()相同
    for i in range(10):
        tofs = pool.apply_async(func, args=(i,))
    pool.close()
    pool.join()
    end = time.time()
    t = end - start
    print('time is :', t)