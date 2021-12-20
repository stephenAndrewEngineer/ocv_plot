#!/usr/bin/env python3
import cv2
import numpy as np

class Axis:
    def __init__(self,img):
        
        # the image is the original cv::mat, which we'll draw into:
        self.img = img
        self.imgWidth = self.img.shape[1]
        self.imgHeight = self.img.shape[0]
        # but we will constrain drawing to a "frame"
        xborder = .1 # border in x-direction, percentage of total frame width
        yborder = .1
        self.frameWidth = int(round((1-xborder)*self.imgWidth))
        self.frameHeight = int(round((1-yborder)*self.imgHeight))
        self.frameLeft = int(round((xborder/2)*self.imgWidth))
        self.frameTop = int(round((yborder/2)*self.imgHeight))
        
        self.xlim = []
        self.ylim = []
        
        # x and y axes:
        self.axis_color = (255,255,255)
        self.axis_thickness = 2
        
        # ticks:
        self.xTickLength = int(.02 * self.imgHeight)
        self.yTickLength = int(.01 * self.imgWidth)
        
    def draw_axes(self):
        xmin = self.xlim[0] - .02 * (self.xlim[1] - self.xlim[0]) 
        xmax = self.xlim[1] + .02 * (self.xlim[1] - self.xlim[0]) 
        ymin = self.ylim[0] - .02 * (self.ylim[1] - self.ylim[0]) 
        ymax = self.ylim[1] + .02 * (self.ylim[1] - self.ylim[0]) 
        xMin,yMin = self.plot_coords_to_img_coords(xmin,ymin)
        xMax,yMax = self.plot_coords_to_img_coords(xmax,ymax)
        x0, y0 = self.plot_coords_to_img_coords(0,0)
        cv2.arrowedLine(self.img,(xMin,y0),(xMax,y0), self.axis_color, self.axis_thickness, tipLength=.01) 
        cv2.arrowedLine(self.img,(x0,yMin),(x0,yMax), self.axis_color, self.axis_thickness, tipLength=.01) 
    
    
    def plot_coords_to_img_coords(self, xp,yp=None):
        # plot_coords are in the "space" of the plot. img coords are pixels
        # xi = A*xp + C
        # yi = B*xp + D
        if (yp is None):
            # callee passed a list
            yp = xp[1]
            xp = xp[0]
            
        A = self.frameWidth / (self.xlim[1] - self.xlim[0])
        C = self.frameLeft - A * self.xlim[0]
        B = self.frameHeight / (self.ylim[0] - self.ylim[1])
        D = self.frameTop - B*self.ylim[1]
        
        xi = A*xp+C
        yi = B*yp+D
        
        return (int(xi),int(yi))

    def draw_yticks(self):    
        for i in range(0,len(self.yticks)):
            (x,y) = self.plot_coords_to_img_coords(0,self.yticks[i])
            cv2.line(self.img,(x+self.yTickLength,y),(x-self.yTickLength,y),(255,255,255),1)
            txt = '%.02f' % self.yticks[i]
            cv2.putText(self.img,txt,(x-4*self.yTickLength,y),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 
            (0,0,225),1,cv2.LINE_AA)
    
    def draw_xticks(self):
        for i in range(0,len(self.xticks)):
            x,y = self.plot_coords_to_img_coords(self.xticks[i],0)
            cv2.line(self.img,(x,y+self.xTickLength),(x,y-self.xTickLength),(255,255,255),1)
            txt = '%.02f' % self.xticks[i]
            cv2.putText(self.img,txt,(x,y+2*self.xTickLength),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,225),1,cv2.LINE_AA)
    

    def plot_histogram(self,y,nBins,rng=None,is_histogram=False):
        # passing rng as not none and is_histogram is False is not a good idea
        if rng is None:
            rng = [min(y), max(y)]
        if is_histogram:
            hist = y
            dBin = (rng[1]-rng[0])/len(y)
            bin_edges = []
            for i in range(0,len(y)+1):
                bin_edges.append(i*dBin)
        else:
            (hist,bin_edges) = np.histogram(y,nBins,range=rng)
            hist = hist/np.sum(hist)    

        self.xlim = rng
        self.ylim = [0,1]
        self.draw_axes()
        
        # draw boxes:
        for i in range(0,len(bin_edges)-1):
            # plot space coordinates:
            xleft = bin_edges[i]
            ytop = hist[i]
            xright = bin_edges[i+1]
            ybottom = 0
            # image space coordinates:
            x1, y1 = self.plot_coords_to_img_coords(xleft, ytop)
            x2, y2 = self.plot_coords_to_img_coords(xright, ybottom)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0)) # TODO - don't draw the bottom line, as it colors the x-axis
        
        # draw a tick for each bin edge (gets hairy if there's too many bins):
        self.xticks =  bin_edges[1:]
        self.draw_xticks()
        # draw 5 ticks along y:
        self.yticks = np.arange(0.2,1.2,.2)        
        self.draw_yticks()

    def plot_curve(self, x, y):
        self.xlim = [min(x), max(x)]
        if (self.xlim[0] == self.xlim[1]):
            self.xlim[0] = min(x) - 1; self.xlim[1] = min(x) + 1
            dx = 2
        else:
            dx = max(x) - min(x)
        self.ylim = [min(y), max(y)]
        if (self.ylim[0] == self.ylim[1]):
            self.ylim[0] = min(y) - 1; self.ylim[1] = min(y) + 1
            dy = 2
        else:
            dy = max(y) - min(y)
        self.draw_axes()
        
        
        self.xticks = np.arange(min(x),max(x),dx/10)
        self.yticks = np.arange(min(y),max(y),dy/10)
        self.draw_xticks()
        self.draw_yticks()
        
        assert len(x) == len(y)
        xlast, ylast = self.plot_coords_to_img_coords(x[0],y[0])
        for i in range(0,len(x)-1):
            xi, yi = self.plot_coords_to_img_coords(x[i+1],y[i+1]) # TODO - precompute
            cv2.line(self.img,(xlast,ylast),(xi,yi),(0,255,0),1,cv2.LINE_AA)
            xlast, ylast = self.plot_coords_to_img_coords(x[i],y[i])
        
    def clear(self):
       self.img[:,:,:] = 0

if __name__ == "__main__":
    imSize = (1280,720)
    img = np.zeros((imSize[1],imSize[0],3),dtype=np.uint8)
    ax = Axis(img)
    ax.plot_histogram(np.array([0.94378378, 0, 0, 0, 0,0, 0, 0, 0, 0.05621622]), 10, (0,255), True)
    
    cv2.imshow('img',img)
    cv2.waitKey(1000)
    
    ax.clear()
    t = np.arange(0,1,1/1000)
    theta = 0
    f = 10
    x = np.cos(2*np.pi * f*t + theta)
    ax.plot_curve(t,x)
    cv2.imshow('img',ax.img)
    cv2.waitKey(1000)
    
    # neat little sinusoidal animation:
    theta = 0
    for i in range(0,400):
        ax.clear()
        theta += 10*np.pi/180
        x = np.cos(2*np.pi * f*t - theta)
        ax.plot_curve(t,x)
        cv2.imshow('img',ax.img)
        cv2.waitKey(1)
    for i in range(0,400):
        ax.clear()
        theta += 5*np.pi/180
        x = np.cos(2*np.pi * f*t - theta)
        ax.plot_curve(t,x)
        cv2.imshow('img',ax.img)
        cv2.waitKey(1)
    for i in range(0,400):
        ax.clear()
        theta += np.pi/180
        x = np.cos(2*np.pi * f*t - theta)
        ax.plot_curve(t,x)
        cv2.imshow('img',ax.img)
        cv2.waitKey(1)
    for i in range(0,400):
        ax.clear()
        theta += np.pi/180
        x = np.cos(2*np.pi * f*t + theta)
        ax.plot_curve(t,x)
        cv2.imshow('img',ax.img)
        cv2.waitKey(1)
    for i in range(0,400):
        ax.clear()
        theta += 5*np.pi/180
        x = np.cos(2*np.pi * f*t + theta)
        ax.plot_curve(t,x)
        cv2.imshow('img',ax.img)
        cv2.waitKey(1)
    for i in range(0,400):
        ax.clear()
        theta += 10*np.pi/180
        x = np.cos(2*np.pi * f*t + theta)
        ax.plot_curve(t,x)
        cv2.imshow('img',ax.img)
        cv2.waitKey(1)
        
