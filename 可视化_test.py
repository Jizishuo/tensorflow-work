# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:53:52 2018

@author: User
"""
import tkinter as tk
class APP:
    def __init__(self,name):
        ##做2个框架。
        frame=[]
        for i in range(2):
            frame.append(tk.Frame(name)) 
        frame[0].pack(side=tk.LEFT,padx=10,pady=3)  
        name.title("yue")        
        self.hi_there = tk.Button(frame[0],text="say hi~",bg="black",fg="white",command=self.say_hi)
        self.hi_there.pack() 
        ##还不能用~
        self.label=tk.Label(name,text="lalala!")
        self.label.pack()

    def say_hi(self):
        print("hello ~ !")


root = tk.Tk()
app = APP(root)
root.mainloop()