# Import required packages
import os
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

PATH = os.getcwd()

IMAGE_PATH = "Images"
THRESHOLD_PATH = "Threshold"
PI = 3.141592653589793

# ORIGINAL = 'A.tif'
# CROPPED = 'A C.png'
# THRESHOLD = 'A T.png'
# OUTPUT_FILENAME = 'A Output.png'
# POSITION_FILENAME = 'A Positions.csv'
# MIN_SIZE = 290
# SCALEBAR_MAX = 1473
# SCALEBAR_MIN = 1105
# SCALEBAR_DISTANCE = 1
# SCATTER_LOCATION = 'A sk.csv'
# FINAL_IMG_NAME = "A SK.png"

# ORIGINAL = 'B.tif'
# CROPPED = 'B C.png'
# THRESHOLD = 'B T.png'
# OUTPUT_FILENAME = 'B Output.png'
# POSITION_FILENAME = 'B Positions.csv'
# MIN_SIZE = 290
# SCALEBAR_MAX = 1463
# SCALEBAR_MIN = 1095
# SCALEBAR_DISTANCE = 1
# SCATTER_LOCATION = 'B sk.csv'
# FINAL_IMG_NAME = "B SK.png"

ORIGINAL = 'D.tif'
CROPPED = 'D C.tif'
THRESHOLD = 'D T.png'
OUTPUT_FILENAME = 'D Output.png'
POSITION_FILENAME = 'D Positions.csv'
MIN_SIZE = 290
SCALEBAR_MAX = 1463
SCALEBAR_MIN = 1091
SCALEBAR_DISTANCE = 1
SCATTER_LOCATION = 'D sk.csv'
FINAL_IMG_NAME = "D SK.png"

def import_image(directory: str, folder: str, image_name: str) -> np.ndarray:
    temp_path = os.path.join(directory, folder, image_name)
    image = cv2.imread(temp_path, cv2.IMREAD_COLOR)
    # cv2.imshow("", image)
    # cv2.waitKey()
    return image

def clean_particles(img: np.ndarray, min_size: int,filename: str) -> np.ndarray:
    img = np.invert(img)
    gray_threshold = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    open_img = ndimage.binary_opening(gray_threshold).astype(int)
    close_img = ndimage.binary_closing(open_img).astype(int)
    no_hole = ndimage.binary_fill_holes(close_img).astype(int)
    arr = no_hole > 0
    output = morphology.remove_small_objects(arr, min_size)
    output = output*255
    output = output.astype(np.uint8)
    output = np.invert(output)
    # cv2.imshow("Cleaned", output)
    # cv2.waitKey()
    plt.imsave(filename, output, cmap=plt.cm.gray)
    return output

def import_clean(filename) ->np.ndarray:
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    return image

def find_contours(image: np.ndarray, position_filename: str):
    gray_cleaned = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_cleaned_blurred = cv2.blur(gray_cleaned, (3, 3))
    contours, hierarchies = cv2.findContours(gray_cleaned_blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(gray_cleaned_blurred.shape[:2],dtype='uint8')
    cv2.drawContours(blank, contours, -1,(255, 0, 0), 1)
    x_pos = np.zeros(len(contours))
    y_pos = np.zeros(len(contours))
    counter = 0
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 1, (0, 0, 255), -1)
        x_pos[counter] = cx
        y_pos[counter] = cy
        counter+=1
    positions = np.dstack((x_pos,y_pos))
    pd.DataFrame(positions[0]).to_csv("positions.csv", index = False, header = False)
    # cv2.imshow("Contours",clean_image);
    # cv2.waitKey();
    return x_pos, y_pos

def calculate_sf(image,x,y,scatter_location):
    Nx = image.shape[1]
    Ny = image.shape[0]
    
    Lx = np.max(x)
    Ly = np.max(y)

    dx = 1/Lx
    dy = 1/Ly

    numx = int(0.2//dx)
    numy = int(0.2//dy)

    qx = np.linspace(-0.1,0.1,num = numx+1)
    qy = np.linspace(-0.1,0.1,num = numy+1)

    S = np.full((numx+1,numy+1), 0.0)

    for i in range(0,len(qx)):
        for j in range(0,len(qy)):
            Sc = (np.sum(np.cos(2*PI*(qx[i]*x_pos + qy[j]*y_pos))))**2
            Ss = (np.sum(np.sin(2*PI*(qx[i]*x_pos + qy[j]*y_pos))))**2
            
            S[i,j] = Sc+Ss

    Np = len(x_pos)
    S = S/(Np)
    S = np.transpose(S)
    dr = np.min([np.min(dx),np.min(dy)])

    rmax = np.min([np.max(qx),np.max(qy)])

    r = np.linspace(0,rmax,(int(rmax//dr)))

    a, b = np.meshgrid(qy, qx)

    R = (np.square(a)+np.square(b))**(1/2)

    pr = np.asarray([])
    for k in range(1,(len(r))):
        mask  = np.logical_and(r[k-1]<R,R<r[k])
        mask = np.transpose(mask)
        values = S[np.where(mask)]
        pr = np.append(pr,np.mean(values))

    # plt.figure(figsize=(4,4), dpi=100);
    # plt.scatter(r[1:],pr);
    # plt.xlim(0,0.1);
    # plt.ylim(0,2);
    # plt.xlabel(r'$k/2\pi$');
    # plt.ylabel("S(k)");
    # plt.show();
    s_k_out = np.dstack((r[1:],pr))
    pd.DataFrame(s_k_out[0]).to_csv(scatter_location, index = False, header = False)
    
    return S

def plot_2d_sf(S,max,min,distance,img):
    scalebar = max-min

    s = distance/scalebar
    
    print(np.shape(S))
    
    x_ticks = np.asarray([0,77,153,230,305])
    # x_ticks = np.asarray([0,62,124,186,246])
    x_tick_labels = np.asarray([-0.1,-0.05,0,0.05,0.1])
    y_ticks = np.asarray([0,50,100,150,200])
    y_tick_labels =np.asarray([0.1,0.05,0,-0.05,-0.1])

    x_tick_labels = x_tick_labels/s
    x_tick_labels = (np.round((x_tick_labels),3))
    x_tick_labels = np.around(x_tick_labels/0.5)*0.5
    y_tick_labels = y_tick_labels/s
    y_tick_labels = (np.round((y_tick_labels),3))
    y_tick_labels = np.around(y_tick_labels/0.5)*0.5
    
    plt.figure(figsize=(4,4), dpi=600);
    custom_colour_map = 'inferno'
    ax = sns.heatmap(S,cmap = custom_colour_map, vmin = 0, vmax = 10, cbar=False)
    ax.set_aspect(S.shape[1]/S.shape[0])
    
    ax.set( xlabel = r'$k_x/2\pi$'+" ("+r'$\mu$'+"m"+ r'$^{-1}$'+")", ylabel = r'$k_y/2\pi$'+" ("+r'$\mu$'+"m"+ r'$^{-1}$'+")")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation = 0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1) 
    cbar = plt.colorbar(ax.collections[0], cax=cax)
    cbar.outline.set_edgecolor('black')
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.savefig(img);
    plt.show()
    

original = import_image(PATH, IMAGE_PATH, ORIGINAL)
cropped = import_image(PATH, IMAGE_PATH, CROPPED)
threshold = import_image(PATH, THRESHOLD_PATH, THRESHOLD)
clean_image = clean_particles(threshold, MIN_SIZE,OUTPUT_FILENAME)
clean_image = import_clean(OUTPUT_FILENAME)
x_pos, y_pos = find_contours(clean_image,POSITION_FILENAME)
S = calculate_sf(clean_image,x_pos,y_pos,SCATTER_LOCATION)
plot_2d_sf(S,SCALEBAR_MAX,SCALEBAR_MIN,SCALEBAR_DISTANCE,FINAL_IMG_NAME)