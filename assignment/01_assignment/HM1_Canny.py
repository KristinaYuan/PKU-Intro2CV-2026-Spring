import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad, x_grad) * 180 / np.pi
    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   

    grad_dir = np.mod(grad_dir, 180)
    mask0 = (grad_dir <= 22.5) | (grad_dir > 157.5)
    mask45 = ((grad_dir > 22.5) & (grad_dir <= 67.5))
    mask90 = ((grad_dir > 67.5) & (grad_dir <= 112.5))
    mask135 = ((grad_dir > 112.5) & (grad_dir <= 157.5))
    NMS_output = np.zeros_like(grad_mag)
    left = np.roll(grad_mag, 1, axis=1)
    right = np.roll(grad_mag, -1, axis=1)
    up = np.roll(grad_mag, 1, axis=0)
    down = np.roll(grad_mag, -1, axis=0)
    up_right = np.roll(up, -1, axis=1)
    up_left = np.roll(up, 1, axis=1)
    down_right = np.roll(down, -1, axis=1)
    down_left = np.roll(down, 1, axis=1)
    NMS_output[mask0 & (grad_mag >= left) & (grad_mag >= right)] = grad_mag[mask0 & (grad_mag >= left) & (grad_mag >= right)]
    NMS_output[mask135 & (grad_mag >= up_right) & (grad_mag >= down_left)] = grad_mag[mask135 & (grad_mag >= up_right) & (grad_mag >= down_left)]
    NMS_output[mask90 & (grad_mag >= up) & (grad_mag >= down)] = grad_mag[mask90 & (grad_mag >= up) & (grad_mag >= down)]
    NMS_output[mask45 & (grad_mag >= up_left) & (grad_mag >= down_right)] = grad_mag[mask45 & (grad_mag >= up_left) & (grad_mag >= down_right)]
    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.20
    high_ratio = 0.40
    average = np.mean(img[img > 0.3])
    maxVal, minVal = average * high_ratio, average * low_ratio
    strong = (img >= maxVal)
    weak = (img >= minVal) & (img < maxVal)
    output = np.zeros_like(img)
    output[strong] = 1
    stack = list(zip(*np.where(strong)))
    h, w = img.shape
    neighbor = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    visited = np.zeros_like(weak, dtype=bool)
    visited[strong] = True
    while stack:
        i, j = stack.pop()
        for u, v in neighbor:
            m, n = i + u, j + v
            if 0 <= m < h and 0 <= n < w and not visited[m, n] and weak[m, n]:
                visited[m, n] = True
                output[m, n] = 1
                stack.append((m, n))
    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
