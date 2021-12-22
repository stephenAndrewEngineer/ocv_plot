#!/usr/bin/env python3
import cv2
import numpy as np

class Axis:
    def __init__(self,img,xborder=.1,yborder=.1):
        
        # the image is the original cv::mat, which we'll draw into:
        self.img = img
        self.imgWidth = self.img.shape[1]
        self.imgHeight = self.img.shape[0]
        # but we will constrain drawing to a "frame"
        #xborder = .1 # border in x-direction, percentage of total frame width
        #yborder = .1
        self.frameWidth = int(round((1-xborder)*self.imgWidth))
        self.frameHeight = int(round((1-yborder)*self.imgHeight))
        self.frameLeft = int(round((xborder/2)*self.imgWidth))
        self.frameTop = int(round((yborder/2)*self.imgHeight))
        
        self.bgColor = (0,0,0)
        
        self.xlim = []
        self.ylim = []
        
        # x and y axes:
        self.axisColor = (255,255,255)
        self.axisThickness = 2
        self.xLabel = ''
        self.yLabel = ''
        
        # ticks:
        self.xTickLength = int(.02 * self.imgHeight)
        self.yTickLength = int(.01 * self.imgWidth)
        self.xTickFormat = '%.02f'
        self.yTickFormat = '%.02f'
        
        self.legend = {'labels':[], 'colors' : [], 'txtWidth' : 0, 'fontFace' : cv2.FONT_HERSHEY_TRIPLEX, 'fontScale' : .45, 'thickness' : 1}
        self.title = ''
                
    def draw_axes(self):
        xmin = self.xlim[0] - .02 * (self.xlim[1] - self.xlim[0]) 
        xmax = self.xlim[1] + .02 * (self.xlim[1] - self.xlim[0]) 
        ymin = self.ylim[0] - .02 * (self.ylim[1] - self.ylim[0]) 
        ymax = self.ylim[1] + .02 * (self.ylim[1] - self.ylim[0]) 
        xMin,yMin = self.plot_coords_to_img_coords(xmin,ymin)
        xMax,yMax = self.plot_coords_to_img_coords(xmax,ymax)
        x0, y0 = self.plot_coords_to_img_coords(0,0)
        cv2.arrowedLine(self.img,(xMin,y0),(xMax,y0), self.axisColor, self.axisThickness, tipLength=.01) 
        cv2.arrowedLine(self.img,(x0,yMin),(x0,yMax), self.axisColor, self.axisThickness, tipLength=.01) 
        # put labels:
        (xHalf,yHalf) = self.plot_coords_to_img_coords((self.xlim[1] + self.xlim[0])/2, (self.ylim[0] + self.ylim[1])/2)
        xLabelOffset = 35 # pixels - label goes this many pixels below axis
        cv2.putText(self.img,self.xLabel,(xHalf,y0+xLabelOffset),cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        
        # to render rotated text, we first render it unrotated, then rotate it (obnoxious!)
        (txtSize,baseline) = cv2.getTextSize(self.yLabel, cv2.FONT_HERSHEY_TRIPLEX, 0.45, 1)
        if txtSize[0] <= 2:
            return
        txtWidth = txtSize[0]
        txtHeight = 2*txtSize[1] + baseline
        #txtHeight = txtSize[1]
        
        # make both even:
        if txtWidth % 2:
            txtWidth += 1
        if txtHeight % 2:
            txtHeight += 1
        
        mat = np.zeros((txtHeight,txtWidth,3),dtype=np.uint8)
        mat[:,:,0] = self.bgColor[0]; mat[:,:,1] = self.bgColor[1]; mat[:,:,2] = self.bgColor[2]
        txtOrg = (0,int(txtHeight/2))
        cv2.putText(mat, self.yLabel, txtOrg, cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255,255,255),1,cv2.LINE_AA)
        newMat = cv2.rotate(mat, cv2.ROTATE_90_COUNTERCLOCKWISE)
        yLabelOffset = 42 # pixels - label goes this many pixels left of axis
        
        bottom = int(yHalf + round(txtWidth/2))
        top = int(yHalf - round(txtWidth/2))
        left = int((x0 - yLabelOffset) - round(txtHeight/2))
        right = int((x0 - yLabelOffset) + round(txtHeight/2))
        #import pdb; pdb.set_trace()
        self.img[top:bottom,left:right,:] = newMat # dangerous - do some bounds checking
    
    
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
            #txt = '%.02f' % self.yticks[i]
            txt = self.yTickFormat % self.yticks[i]
            cv2.putText(self.img,txt,(x-4*self.yTickLength,y),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 
            (0,0,225),1,cv2.LINE_AA)
    
    def draw_xticks(self):
        for i in range(0,len(self.xticks)):
            x,y = self.plot_coords_to_img_coords(self.xticks[i],0)
            cv2.line(self.img,(x,y+self.xTickLength),(x,y-self.xTickLength),(255,255,255),1)
            #txt = '%.02f' % self.xticks[i]
            txt = self.xTickFormat % self.xticks[i]
            cv2.putText(self.img,txt,(x,y+2*self.xTickLength),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,225),1,cv2.LINE_AA)
    
    def set_legend(self, *curves):
        # curves should be label, color, label, color, ...
        self.legend['labels'] = []
        self.legend['colors'] = []
        for i in range(0,len(curves),2):
            self.legend['labels'].append(curves[i])
            self.legend['colors'].append(curves[i+1])

        self.legend['txtWidth'] = 0
        for i in self.legend['labels']:
            (txtSize, _) = cv2.getTextSize(i,self.legend['fontFace'],self.legend['fontScale'], self.legend['thickness'])
            self.legend['txtWidth'] = max(self.legend['txtWidth'],txtSize[0])
            

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

    def plot_curve(self, x, y,color=(0,255,0)):
        self.set_scaling_for_curves([[x,y]])
        self.plot_curve_raw(x,y)
    
    def set_scaling_for_curves(self, curves):
        # each element in curves should be a two-element list (or tuple) - first is x, then y:
        x = np.zeros(()); y = np.zeros(())
        for i in curves:
            x = np.hstack((x,i[0]))
            y = np.hstack((y,i[1]))
        #import pdb; pdb.set_trace()
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
        
    def plot_curve_raw(self,x,y,color=(0,255,0)):    
        
        assert len(x) == len(y)
        xlast, ylast = self.plot_coords_to_img_coords(x[0],y[0])
        for i in range(0,len(x)-1):
            if (np.isnan(y[i+1])):
                continue
            xi, yi = self.plot_coords_to_img_coords(x[i+1],y[i+1]) # TODO - precompute
            cv2.line(self.img,(xlast,ylast),(xi,yi),color,1,cv2.LINE_AA)
            xlast, ylast = self.plot_coords_to_img_coords(x[i],y[i])
        self.draw_legend()
        self.draw_title()
           
    def draw_legend(self):
        if len(self.legend['labels']):
            # get size:
            leftPad = 10
            width = self.legend['txtWidth'] + 80
            height = 30 * len(self.legend['labels'])
            (right,top) = self.plot_coords_to_img_coords(.98 * (self.xlim[1]-self.xlim[0]) + self.xlim[0], 
                                                         .98 * (self.ylim[1]-self.ylim[0]) + self.ylim[0])
            left = right - width
            bottom = top + height
            # draw a box:
            cv2.rectangle(self.img, (int(left),int(top)), (int(right), int(bottom)), (255,255,255), 1)
            for i in range(0,len(self.legend['labels'])):
                cv2.putText(self.img, self.legend['labels'][i],(int(left+leftPad), int(top+20)),self.legend['fontFace'],self.legend['fontScale'],
                (255,255,255), self.legend['thickness'],cv2.LINE_AA)
                lineLeft = int(left+leftPad + self.legend['txtWidth'] + 10)
                lineRight = int(lineLeft + 20)
                lineY = int(top+15)
                cv2.line(self.img, (lineLeft, lineY),(lineRight, lineY), self.legend['colors'][i], 1, cv2.LINE_AA)
                top += 20
            
    def draw_title(self):
        txtPos = self.plot_coords_to_img_coords(.3 * (self.xlim[1]-self.xlim[0]) + self.xlim[0],
                                                .98 * (self.ylim[1]-self.ylim[0]) + self.ylim[0])
        cv2.putText(self.img,self.title,txtPos, cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255,255,255),1,cv2.LINE_AA)
            
    def clear(self):
       self.img[:,:,0] = self.bgColor[0]
       self.img[:,:,1] = self.bgColor[1]
       self.img[:,:,2] = self.bgColor[2]

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
        
