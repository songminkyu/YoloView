#!/usr/bin/env python3
import cv2
import numpy as np
import math
import json

import tkinter as tk
from tkinter import ttk
import threading
import queue
from ping3 import ping
import re

# --- Zoom in to chosen CCTV view on Left Click mouse event ---
def Mouse_Event(event, x, y, flags, param):
    global view

    if event == cv2.EVENT_LBUTTONDOWN:
       if(view[3]):
         view[3]=False
       else:
         col=int(x/view[1])
         row=int(y/view[2])
         view[3]=[col,row]
         print("View: ",col, row)
    elif event == cv2.EVENT_RBUTTONDOWN:
         view[3]=False

# --- Open RTSP stream and update img global variable ---
def openvideo(strm,y):
    global img

    x = re.search("rtsp:\/\/(\d+[.]\d+[.]\d+[.]\d+)", strm)

    # --- Check for valid IP address ---
    if(x):
      host=x[1]
               
      for i in range(0,5):
       if(ping(host)):
        img[y]=cv2.VideoCapture(strm, cv2.CAP_FFMPEG) #Open RTSP stream if pingable
        break
       elif(i>3):
        img[y]="IP address not reachable"
        print("ERROR: cannot ping "+ host)
        break
    else:
      img[y]="Invalid IP address"

# --- Show CCTV video stream on location drop down box chosen event ---
def callbackFunc(event):
  global comboExample, jdata, img, screen_width, screen_height, view 

  print(comboExample.current(), comboExample.get())

  stream=[]
  view=[]

  #-- Load RSTP url from the chosen location
  stream = jdata[comboExample.get()]
  print(stream)

  #-- Calculate the multi-camera view
  camnum= len(stream)
  print("Number of cameras: "+str(camnum))
  #view.append(int(math.sqrt(camnum)))
  view.append(math.ceil(math.sqrt(camnum)))
  
  view.append(int(screen_width/view[0]))
  view.append(int(screen_height/view[0]))
  print("View: "+str(view[0])+":"+str(view[0])+" ,Rowsize: "+str(view[1])+";"+str(view[2]))
  view.append(False) #initialization for a single view window

  #-- Start multiple thread of Openvideo()
  img={}
  p=[]
  for i in range(0, camnum):
    img[i]="Waiting.."
    p.append(threading.Thread(target=openvideo, args=(stream[i],i)))
    p[i].start()

  for i in range(camnum, view[0]*view[0]):
    img[i]="No Camera"

  # --- display blank CV2 screen to bind to mouse event ---
  blankscrn = np.zeros((screen_height,screen_width, 3), dtype=np.uint8)
  cv2.imshow('Multi-camera View', blankscrn)
  cv2.setMouseCallback('Multi-camera View', Mouse_Event)

  # --- Multi-camera View window is not closed, display CCTV RTSP video stream ---
  while(int(cv2.getWindowProperty('Multi-camera View', cv2.WND_PROP_VISIBLE ))>0):
        showimg(img, view)

        # exit if pressed any key
        # (it doesn't wait for key so you can read next frame)
        # (you need opened window to catch pressed key)

        if cv2.waitKey(1) != -1:
            break

  # close stream after Multi-camera View window is closed
  for i in range(0, camnum):
    if(img[i]!="Invalid IP address" and img[i]!="IP address not reachable"
       and img[i]!="Waiting.." and img[i]!="No Camera"):
      img[i].release()

  # close window
  cv2.destroyAllWindows()   

# --- Read videos from RTSP stream and display inside CV2 window
def showimg(img, view):
  global resizedframe
  if(view[3]): #single view mode
      addvideo(view[3][1], view[3][0], img, view)
      Verti = resizedframe[view[3][0]]
  else:  #multi-view mode
    videorow= []
    for i in range(0,view[0]):
      Verti=videorow.append(addvideorow(i, img, view))
 
    # concatenate image Vertically
    for i in range(0,view[0]):
          hori=[]
          for x in range(0, view[0]):
            hori.append(np.concatenate((videorow[x]), axis=1))
          if(i==0):
            multi=(hori[i],)
          else:
            multi+=(hori[i],)
 
    Verti = np.concatenate(multi, axis=0)

  # (open window and) display one frame of multiple cameras
  cv2.imshow('Multi-camera View', Verti)

# --- Read one RTSP frame
def addvideo(i,x, img, view):
    global resizedframe
    readerr=[False, ""]

    # check whether video stream is opened  
    if(img[i*view[0]+x]=="Invalid IP address" or img[i*view[0]+x]=="Waiting.."
       or img[i*view[0]+x]=="IP address not reachable" or img[i*view[0]+x]=="No Camera"):
          readerr=[True, img[i*view[0]+x]]
    else:  
            if img[i*view[0]+x].isOpened():
              # read one frame (and record error in readerr)
              ret, readframe = img[i*view[0]+x].read()
              if not ret:
                readerr=[True, "Stream error"]
              else:
                # resize video so all of them fit inside the screen
                if(view[3]):                
                  resizedframe[x] = readframe
                else:
                  resizedframe[x] = cv2.resize(readframe, (view[1], view[2]))
            else:
              readerr=[True, "Unable to open CCTV stream"]
            
      
    if(readerr[0]):
              resizedframe[x] = np.zeros((view[2], view[1],3), dtype=np.uint8)
              # Write error message for this video on CV2 window
              resizedframe[x]  = cv2.putText(resizedframe[x], readerr[1], (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)

    

# --- Read mulitple RTSP frame one row at a time
def addvideorow(i, img, view):
    global resizedframe

    resizedframe={}
    t=[]
    # --- Start multiple thread to read individual video for one row
    for x in range(0, view[0]):
      t.append(threading.Thread(target=addvideo, args=(i,x,img, view)))
      t[x].start()

    for x in range(0, view[0]):
      t[x].join()
      #add video to a horizontal row
      if(x==0):
              temprow=(resizedframe[x],)
      else:
              temprow+=(resizedframe[x],)
    return temprow
    
# --- Main Loop
if __name__ == "__main__":

    global comboExample, jdata, screen_width, screen_height

    #load cameras RTSP stream url from JSON file
    with open('camera.json', 'r') as f:
      jdata = json.load(f)

    #Start Tk GUI
    app = tk.Tk()
    app.geometry('310x50')
    frame1 = tk.Frame(app)

    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    print("screensize: "+str(screen_width)+";"+str(screen_height))

    #load cameras from JSON file for drop down box
    location=[]
    for item in jdata:
      location.append(item)
      
    #Define a drop down box for the CCTV locations defined in JSON file
    labelTop = tk.Label(app,
                    text = "Choose location ")
    labelTop.grid(column=0, row=1, padx=8, pady=2)

    comboExample = ttk.Combobox(app, 
                            values=location)
    comboExample.grid(column=1, row=1, sticky=tk.W)
    labelbottom = tk.Label(app,
                    text='Mouse click to zoom in to individual CCTV')
    labelbottom.grid(column=0, row=2, sticky=tk.W, columnspan=2, padx=8, pady=2)

    comboExample.bind("<<ComboboxSelected>>", callbackFunc)

    app.mainloop()

