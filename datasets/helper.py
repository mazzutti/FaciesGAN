import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import re
import torch
import os
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker

def RGBToImage(rgbarray:np.ndarray):
    """
    Creates a new image with the given mode and size.
    :param array: ndarray shape with (width, height,3)
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    size=(rgbarray.shape[0],rgbarray.shape[1])
    im = Image.new('RGB',size)
    for i in range(size[0]):
        for j in range(size[1]):
            im.putpixel((i,j),(rgbarray[i,j,0],rgbarray[i,j,1],rgbarray[i,j,2]))
    return im

def GrayToImage(rgbarray:np.ndarray):
    """
    Creates a new image with the given mode and size.
    :param array: ndarray shape with (width, height,1)
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    size=(rgbarray.shape[0],rgbarray.shape[1])
    im = Image.new('L',size)
    for i in range(size[0]):
        for j in range(size[1]):
            im.putpixel((i,j),int(rgbarray[i,j]))
    return im

def ArrayToRGB(array:np.ndarray,dict:dict[int,np.array]):
    """
    Creates a new image array with the given map relations
    :param array: array shape with (width, height)
    :param dict: array value map to the rbg,
    the map value is array with shape (3) represent rgb,
    :returns: array shape with (width, height,3) default rbg is (255,255,255)
    """
    rgbarray=np.full(shape=(array.shape[0],array.shape[1],3), fill_value=0,dtype=int)
    if(np.size(array.shape)>2):
        assert array.shape[2]==1
        array=np.squeeze(array, 2)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if(array[i,j] in dict):
                rgbarray[i,j,:]=dict[array[i,j]]
            else:
                rgbarray[i,j,:]=[255,255,255]
    return rgbarray

def ArrayToGray(array:np.ndarray,dict:dict[int,int]):
    """
    Creates a new image array with the given map relations
    :param array: array shape with (width, height)
    :param dict: array value map to the gray,
    :returns: array shape with (width, height,1)
    """
    grayarray=np.full(shape=(array.shape[0],array.shape[1]), fill_value=0,dtype=int)
    if(np.size(array.shape)>2):
        assert array.shape[2]==1
        array=np.squeeze(array, 2)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if(array[i,j] in dict):
                grayarray[i,j]=dict[array[i,j]]
            else:
                grayarray[i,j]=255
    return grayarray

def WriteGslibFile(grid:np.ndarray,name:str):
    shape = grid.shape
    icount=shape[0]
    jcount=shape[1]
    kcount=1
    if(len(shape)==3):
        kcount=shape[2]
    f1 = open(name,'w')
    f1.write(str(icount))
    f1.write(' ')
    f1.write(str(jcount))
    f1.write(' ')
    f1.write(str(kcount))
    f1.write('\n')
    f1.write('1\n')
    f1.write('v\n')
    for k in range(kcount):
        for j in range(jcount):
            for i in range(icount):
                if(len(shape)==3):
                    f1.write(str(grid[i,j,k]))
                else:
                    f1.write(str(grid[i,j]))
                f1.write('\n')
    f1.flush()
    f1.close()

def OpenGslibFile(name):
    f1 = open(name,'r')
    data = f1.read().splitlines()
    f1.close()
    if('(' in data[0]):
        list=re.split(r'x|\(|\)|\s',data[0])
        gridsize = [int(list[2]),int(list[3]),int(list[4])]
        start = 3
    else:
        list = data[0].split(' ')
        gridsize = [int(list[0]),int(list[1]),int(list[2])]
        start = int(list[len(list) - 1]) * 2 + 1
    ijcount = gridsize[0] * gridsize[1]
    icount = gridsize[0]
    labels = np.full(gridsize,fill_value=-99.0,dtype=np.float32, order='F')
    for k in range(gridsize[2]):
        for j in range(gridsize[1]):
            for i in range(gridsize[0]):
                labels[i,j,k] = float(data[k * ijcount + j * icount + i + start])
    return labels

    channel = mpimg.imread(filename)
    width = channel.shape[1]
    height = channel.shape[0]
    binary = np.ones(shape=(width, height), dtype=np.int32, order='F')
    for i in range(width):
        for j in range(height):
            if channel[j,i,0] >= 0.5:
                binary[i,j] = 0
    return binary

def FakeToImg(fake:np.ndarray):
    arr=fake
    assert(arr.ndim==4)
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            for k in range(0,arr.shape[2]):
                for l in range(0,arr.shape[3]):
                    value = arr[i][j][k][l]
                    if value >= 0:
                        arr[i][j][k][l] = 1
                    else:
                        arr[i][j][k][l] = 0
    return arr

def gen_filter(ijs,vals):
    newijs=[]
    for scale in range(len(ijs)):
        ij=[]
        for point_i in range(len(ijs[scale])):
            ij.append([ijs[scale][point_i][0]  ,ijs[scale][point_i][1]])
            ij.append([ijs[scale][point_i][0]+1,ijs[scale][point_i][1]])
            ij.append([ijs[scale][point_i][0]-1,ijs[scale][point_i][1]])
            ij.append([ijs[scale][point_i][0]  ,ijs[scale][point_i][1]+1])
            ij.append([ijs[scale][point_i][0]  ,ijs[scale][point_i][1]-1])
            ij.append([ijs[scale][point_i][0]  ,ijs[scale][point_i][1]+2])
            ij.append([ijs[scale][point_i][0]  ,ijs[scale][point_i][1]-2])
        newijs.append(ij)
    newvals=[]
    for scale in range(len(vals)):
        newvals.append(vals[scale])
        newvals.append(vals[scale]*0.5)
        newvals.append(vals[scale]*0.5)
        newvals.append(vals[scale]*0.7)
        newvals.append(vals[scale]*0.7)
        newvals.append(vals[scale]*0.3)
        newvals.append(vals[scale]*0.3)
    return np.asarray(newijs),np.asarray(newvals)

def save_tensor_to_gslib(input:torch.Tensor,folder:str,file_pre:str=None,file_names:list=None):
    imgs = input.detach().cpu().numpy()
    for i in range(imgs.shape[0]):
        img=imgs[i]
        if file_names is not None:
            name=os.path.join(folder,file_names[i]) 
        else:
            name=os.path.join(folder,"{0}_{1}.gslib".format(file_pre,i+1))
        binary=(img.squeeze()>0).astype(np.int32)
        WriteGslibFile(binary,name)