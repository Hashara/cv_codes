import cv2
import numpy as np
import os

"""
This will automatically read images in same directory,
 produce the output as "canny_edge_detection_"+<image_name> 
 save in the same directory
"""


def read_image(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    return img.tolist()


def image_write(img, file_name):
    cv2.imwrite(file_name + '.jpg', np.array(img))


def zero_array(x, y):
    """
    :param x: x dim
    :param y: y dim
    :return: 2d list of 0s
    """
    return [[0 for j in range(y)] for i in range(x)]


def one_array(x, y):
    """
    :param x: x dim
    :param y: y dim
    :return: 2d list of 1s
    """
    return [[1 for j in range(y)] for i in range(x)]


def to_gray_scale(img):
    x_shape = len(img)
    y_shape = len(img[0])
    output = zero_array(x_shape, y_shape)

    for i in range(x_shape):
        for j in range(y_shape):
            # 0.2989 * R + 0.5870 * G + 0.1140 * B
            output[i][j] = 0.2989 * img[i][j][0] + 0.5870 * img[i][j][1] + 0.1140 * img[i][j][2]
    return output


def select_range(arr, x1, x2, y1, y2):
    return [i[y1:y2] for i in arr[x1:x2]]


def assign_to_range(arr, assigner, x1, x2, y1, y2):
    x, y, z = 0, 0, 0
    for i in range(x1, x2):
        y = 0
        for j in range(y1, y2):
            arr[j][i] = assigner[y][x]
            y += 1
        x += 1
    return arr


def edge_op(img, filter_size):
    """
    :param filter_size: size of the kernel
    :param img:  list of image ( 1 colour channel)
    :return:  list of image wrapped from edge pixel
    """
    # create a zero filled array for save the output
    edg_img = zero_array(len(img) + (filter_size - 2) * 2,
                         len(img[0]) + (filter_size - 2) * 2)
    # copy image to the middle of the output
    edg_img = assign_to_range(edg_img, img,
                              (filter_size - 2), len(edg_img[0]) - (filter_size - 2),
                              (filter_size - 2), len(edg_img) - (filter_size - 2))
    # wrap edges
    edg_img = assign_to_range(edg_img, img[-(filter_size - 2):],
                              (filter_size - 2), len(edg_img[0]) - (filter_size - 2),
                              0, (filter_size - 2))
    edg_img = assign_to_range(edg_img, img[0: (filter_size - 2)],
                              (filter_size - 2), len(edg_img[0]) - (filter_size - 2),
                              len(edg_img) - (filter_size - 2), len(edg_img))
    edg_img = assign_to_range(edg_img, select_range(img, 0, len(img), len(img[0]) - (filter_size - 2),
                                                    len(img[0])), 0, (filter_size - 2), (filter_size - 2),
                              len(edg_img) - (filter_size - 2))
    edg_img = assign_to_range(edg_img, select_range(img, 0, len(img), 0, (filter_size - 2)),
                              len(edg_img[0]) - (filter_size - 2), len(edg_img[0]), (filter_size - 2),
                              len(edg_img) - (filter_size - 2), )
    return edg_img


def conv_op_sum(x, y):
    res = 0
    for i in range(len(x)):
        for j in range(len(y)):
            res += x[i][j] * y[i][j]
    return res


def gaussian_filter_operation(img, filter_size=3):
    """
    :param img: image
    :param filter_size: kernel size
    :return: filtered image
    """
    kernel = one_array(filter_size, filter_size)
    kernel = [[y / (filter_size ** 2) for y in x] for x in kernel]
    return conv_operation(img, kernel)


def g_x(img):
    """
    Horizontal edges detection by using sobel
    :param img: image
    :return: horizontal edge detected image
    """
    kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    return conv_operation(img, kernel)


def g_y(img):
    """
    Vertical edges detection by using sobel
    :param img: image
    :return: vertical edge detected image
    """
    kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    return conv_operation(img, kernel)


def g(gx, gy):
    """
    magnitude of gradient
    :param gx: horizontal edge image
    :param gy: vertical edge image
    :return: (Gx ** 2 + Gy ** 2) ** (0.5)
    """
    x_shape = len(gx[0])
    y_shape = len(gx)
    output = zero_array(y_shape, x_shape)

    for x in range(x_shape):
        for y in range(y_shape):
            output[y][x] = (gx[y][x] ** 2 + gy[y][x] ** 2) ** (0.5)  # G = (Gx ** 2 + Gy ** 2) ** (.5)
    return output


def find_max(array):
    return max(max(array, key=max))


def non_max_suppression(G, gx, gy):
    """
    :param G: Magnitude of gradient
    :param gx: Gradient in X
    :param gy:Gradient in Y
    :return:non maxima suppressed image
    """
    x_shape = len(gx[0])
    y_shape = len(gx)
    output = zero_array(y_shape, x_shape)

    for x in range(x_shape):
        for y in range(y_shape):
            Gx = gx[y][x]
            Gy = gy[y][x]
            q = 255
            r = 255
            try:
                # Estimate the direction and do non-maxima suppression
                if Gx > 0 and Gy > 0:
                    # angle is in between 0 and 22.5
                    if Gy / Gx <= 0.41421356237:
                        q = G[y + 1][x]
                        r = G[y - 1][x]
                    # angle is in between 22.5 and 67.5
                    elif Gy / Gx <= 2.41421356237:
                        q = G[y - 1][x + 1]
                        r = G[y + 1][x - 1]
                    # angle between 67.57 and 90
                    else:
                        q = G[y][x + 1]
                        r = G[y][x - 1]
                elif Gx < 0 and Gy > 0:
                    if Gy / Gx >= -0.41421356237:
                        q = G[y + 1][x]
                        r = G[y - 1][x]
                    elif Gy / Gx >= -2.41421356237:
                        q = G[y - 1][x - 1]
                        r = G[y + 1][x + 1]
                    else:
                        q = G[y][x + 1]
                        r = G[y][x - 1]
                elif Gx < 0 and Gy < 0:
                    if Gy / Gx <= 0.41421356237:
                        q = G[y + 1][x]
                        r = G[y - 1][x]
                    elif Gy / Gx <= 2.41421356237:
                        q = G[y - 1][x + 1]
                        r = G[y + 1][x - 1]
                    else:
                        q = G[y][x + 1]
                        r = G[y][x - 1]
                elif Gx > 0 and Gy < 0:
                    if Gy / Gx >= -0.41421356237:
                        q = G[y + 1][x]
                        r = G[y - 1][x]
                    elif Gy / Gx >= -2.41421356237:
                        q = G[y - 1][x - 1]
                        r = G[y + 1][x + 1]
                    else:
                        q = G[y][x + 1]
                        r = G[y][x - 1]
                elif Gx == 0:
                    if Gy > 0:
                        q = G[y][x + 1]
                        r = G[y][x - 1]
                    elif Gy < 0:
                        q = G[y][x + 1]
                        r = G[y][x - 1]
                    else:
                        q = G[y + 1][x]
                        r = G[y - 1][x]
                else:
                    q = G[y + 1][x]
                    r = G[y - 1][x]
                if (G[y][x] >= q) and (G[y][x] >= r):
                    output[y][x] = G[y][x]
                else:
                    output[y][x] = 0
            except IndexError as e:
                pass
    return output


def threshold(img, low_ratio=0.05, high_ratio=0.09):
    """
    :param img: image
    :param low_ratio:
    :param high_ratio:
    :return: double threshold image
    """
    # high and low thresholds from ratios
    high_threshold = find_max(img) * high_ratio
    low_threshold = high_threshold * low_ratio

    x_shape = len(img[0])
    y_shape = len(img)
    output = zero_array(y_shape, x_shape)

    weak = 25  # when value is between two thresholds
    strong = 255  # when value is higher than the high threshold

    for x in range(x_shape):
        for y in range(y_shape):
            if img[y][x] >= high_threshold:
                output[y][x] = strong
            elif high_threshold >= img[y][x] >= low_threshold:
                output[y][x] = weak
            else:
                output[y][x] = 0
    return output


def conv_operation(img, kernel):
    kernel = flip(kernel)  # flip the kernel

    filter_size = len(kernel)
    x_shape = len(img[0])
    y_shape = len(img)
    output = zero_array(y_shape, x_shape)

    edge_img = edge_op(img, filter_size)  # do edge operation

    for x in range(x_shape):
        for y in range(y_shape):
            # conv operation
            output[y][x] = (conv_op_sum(kernel, select_range(edge_img, y, y + filter_size, x,
                                                             x + filter_size)))
    return output


def flip(arr):
    arr = arr[::-1]
    for i in range(len(arr)):
        arr[i] = arr[i][::-1]
    return arr


def canny_edge_detect_algo(img):
    """
    :param img: image as a 2d array
    :return: canny edge detected image
    """
    gray = to_gray_scale(img)  # convert gray scale
    gaussian = gaussian_filter_operation(gray)  # gaussian filtering
    gx = g_x(gaussian)  # calculate Gx and Gy
    gy = g_y(gaussian)
    G = g(gx, gy)  # gradient magnitude
    non_max = non_max_suppression(G, gx, gx)  # non maxima suppression and gradient direction
    threshold_image = threshold(non_max)  # double thresholding
    return threshold_image


if __name__ == '__main__':
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        n = f.split('.')
        if n[-1] == 'jpg' or n[-1] == 'jpeg' or \
                n[-1] == 'JPEG' or n[-1] == 'JPG' or \
                n[-1] == "png" or n[-1] == "PNG":
            image = read_image(f)
            out = canny_edge_detect_algo(image)
            image_write(out, 'canny_edge_detection_' + n[0])
