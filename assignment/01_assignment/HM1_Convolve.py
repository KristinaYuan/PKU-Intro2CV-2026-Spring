import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type=="zeroPadding":
        H, W = img.shape
        matrix1 = np.eye(H+2*padding_size, H, k=-padding_size)
        matrix2 = np.eye(W, W+2*padding_size, k=padding_size)
        padding_img = matrix1 @ img @ matrix2
        return padding_img
    elif type=="replicatePadding":
        H, W = img.shape
        matrix1 = np.eye(H+2*padding_size, H, k=-padding_size)
        matrix2 = np.eye(W, W+2*padding_size, k=padding_size)
        matrix1[:padding_size, 0] = 1
        matrix1[-padding_size:, -1] = 1
        matrix2[0, :padding_size] = 1
        matrix2[-1, -padding_size:] = 1
        padding_img = matrix1 @ img @ matrix2
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img = padding(img, 1, "zeroPadding")

    #build the Toeplitz matrix and compute convolution
    img_flat = padding_img.reshape(-1)
    toeplitz = np.zeros((36,64))
    i, j = np.meshgrid(np.arange(6), np.arange(6))
    u, v = np.meshgrid(np.arange(3), np.arange(3))
    u = u.reshape(-1)
    v = v.reshape(-1)
    r = (i.reshape(-1,1) + u.reshape(1,-1))
    c = (j.reshape(-1,1) + v.reshape(1,-1))
    indices = 8 * r + c
    kernel_flat = kernel.reshape(-1)
    toeplitz[np.arange(36)[:,None], indices] = kernel_flat[None,:]
    output = toeplitz @ img_flat
    output = output.reshape(6,6)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    H, W = img.shape
    k = kernel.shape[0]
    Hout, Wout = H-k+1, W-k+1
    i,j = i, j = np.meshgrid(np.arange(Hout), np.arange(Wout),indexing='ij')
    i = i.reshape(-1,1)
    j = j.reshape(-1,1)
    u, v = np.meshgrid(np.arange(k), np.arange(k),indexing='ij')
    u = u.reshape(1,-1)
    v = v.reshape(1,-1)
    r = i + u
    c = j + v
    patch = img[r,c]
    kernel_flat = kernel.reshape(-1)
    output = (patch @ kernel_flat).reshape(Hout,Wout)
    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "zeroPadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("Lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)


    