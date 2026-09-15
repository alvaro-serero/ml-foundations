"""Simple Convolutional 2D Layer"""

import numpy as np


def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    # Get kernel shapes
    kernel_height, kernel_width = kernel.shape

    # Pad the input with zeros + get padded input shapes
    input_padded = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)
    input_height, input_width = input_padded.shape

    # Calculate output dimensions
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    # Create empty output matrix
    output = np.zeros((output_height, output_width))

    # Convolution
    for i in range(output_height):
        for j in range(output_width):
            row_start = i * stride
            col_start = j * stride
            row_end = row_start + kernel_height
            col_end = col_start + kernel_width
            window = input_padded[row_start:row_end, col_start:col_end]
            output[i, j] = np.sum(window * kernel)

    return output


input_matrix = np.array([
    [1., 2., 3., 4., 5.],
    [6., 7., 8., 9., 10.],
    [11., 12., 13., 14., 15.],
    [16., 17., 18., 19., 20.],
    [21., 22., 23., 24., 25.],
])
kernel = np.array([
    [1., 2.],
    [3., -1.],
])
padding, stride = 0, 1
expected = np.array([
    [ 16., 21., 26., 31.],
    [ 41., 46., 51., 56.],
    [ 66., 71., 76., 81.],
    [ 91., 96., 101., 106.],
])
output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
