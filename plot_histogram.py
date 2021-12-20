#!/usr/bin/env python3

# note - I am not sure this will work if rng[0] is not 0

import numpy as np
import cv2


def plot_histogram(img,y,nBins,rng,is_histogram=False):
    if is_histogram:
        hist = y
        dBin = (rng[1]-rng[0])/len(y)
        bin_edges = []
        for i in range(0,len(y)+1):
            bin_edges.append(i*dBin)
    else:
        (hist,bin_edges) = np.histogram(y,nBins,range=rng)
        hist = hist/np.sum(hist)
    
    imWidth = img.shape[1]
    imHeight = img.shape[0]
    
    
    # origin:
    x0 = int(0.1 * imWidth)
    y0 = int(0.9 * imHeight)
    # scaling:
    x_fs = int(0.8 * imWidth)
    y_fs = int(0.8 * imHeight) 
    
    # transformations:
    #A = (bin_edges[-1] - x0) / (x0+x_fs)
    A = x_fs / bin_edges[-1]
    B = -y_fs  # /1
    
    # draw the axes
    # these axes go from 0->1 (y) and rng[0]->rng[1] (x)
    #cv2.arrowedLine(img,(x0,y0),(x0,y0-y_fs),(255,255,255),2,tipLength=.01) # y-axis
    #cv2.arrowedLine(img,(x0,y0),(x0+x_fs,y0),(255,255,255),2,tipLength=.01) # x-axis
    
    # these axes put the arrow 10 % farther (in plot space) then the last highest data point:
    cv2.arrowedLine(img,(x0,y0),(x0,int(B*1.1+y0)),(255,255,255),2,tipLength=.01) # y-axis
    extra = .1 * (rng[1] - rng[0])
    endPoint = extra + rng[1]
    cv2.arrowedLine(img,(x0,y0),(int(x0+A*endPoint),y0),(255,255,255),2,tipLength=.01) # x-axis

    # draw a tick for each bin edge (gets hairy if there's too many bins):
    xTickLength = int(.02 * imHeight)
    for i in range(1,len(bin_edges)):
        #x = int(bin_edges[i]*x_fs+x0)
        x = int(bin_edges[i]*A + x0)
        cv2.line(img,(x,y0+xTickLength),(x,y0-xTickLength),(255,255,255),1)
        txt = '%.02f' % bin_edges[i]
        cv2.putText(img,txt,(x,y0+2*xTickLength),
        cv2.FONT_HERSHEY_TRIPLEX, 0.5, 
        (0,0,225),1,cv2.LINE_AA)
    
    # draw 5 ticks along y:
    yTickLength = int(.01 * imWidth)
    #dy = (rng[1] - rng[0])/5
    dy = 1/5
    for i in range(0,5):
        #y = int(y0 - (i+1)*dy*y_fs)
        y = int(B*(i+1)*dy + y0)
        cv2.line(img,(x0+yTickLength,y),(x0-yTickLength,y),(255,255,255),1)
        txt = '%.02f' % ((i+1)*dy)
        cv2.putText(img,txt,(x0-4*yTickLength,y),
        cv2.FONT_HERSHEY_TRIPLEX, 0.5, 
        (0,0,225),1,cv2.LINE_AA)

    # draw boxes:
    for i in range(0,len(bin_edges)-1):
        # plot space coordinates:
        xleft = bin_edges[i]
        ytop = hist[i]
        xright = bin_edges[i+1]
        ybottom = 0
        # image space coordinates:
        x1 = int(xleft*A + x0)
        x2 = int(xright*A + x0)
        y1 = int(ytop*B + y0)
        y2 = int(ybottom*B + y0)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))


if __name__=="__main__":
   
   nBins = 20
   imSize = (1280,720)
   #imSize = (640,360)
   y = np.random.rand(100)
   rng = (0,1)
   img = np.zeros((imSize[1],imSize[0],3),dtype=np.uint8)
   plot_histogram(img,y,nBins,rng)

   cv2.imshow('histogram',img)
   cv2.waitKey(0)
   
   hist = np.array([0.94378378, 0, 0, 0, 0,0, 0, 0, 0, 0.05621622])
   imSize = (1280,720)
   nBins = 10
   rng = (0,255)
   img = np.zeros((imSize[1],imSize[0],3),dtype=np.uint8)
   plot_histogram(img,hist,nBins,rng,True)

   cv2.imshow('histogram',img)
   cv2.waitKey(0)
   