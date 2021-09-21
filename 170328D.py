'''
numpy and opencv was used only for reading and saving images
'''

import cv2
import numpy as np
import os


def read_image(file_name):
    '''
    Read the image when the file name given
    :param file_name: file name
    :return: image as a list
    '''
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # return image as a list
    return image.tolist()


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge_operation_2d(image, filter_size):
    '''
    2D image (1 colour channel) wrapping from the edge pixel
    :param image:  list of image ( 1 colour channel)
    :return:  list of image wrapped from edge pixel
    '''
    # create a zero filled array for save the output
    edge_processed_image = generate_zero_array_2d(len(image) + (filter_size - 2) * 2,
                                                  len(image[0]) + (filter_size - 2) * 2)

    # copy image to the middle of the output
    edge_processed_image = assign_to_range(edge_processed_image, image,
                                           (filter_size - 2), len(edge_processed_image[0]) - (filter_size - 2),
                                           (filter_size - 2), len(edge_processed_image) - (filter_size - 2))

    # wrap edges
    edge_processed_image = assign_to_range(edge_processed_image, image[-(filter_size - 2):],
                                           (filter_size - 2), len(edge_processed_image[0]) - (filter_size - 2),
                                           0, (filter_size - 2))

    edge_processed_image = assign_to_range(edge_processed_image, image[0: (filter_size - 2)],
                                           (filter_size - 2), len(edge_processed_image[0]) - (filter_size - 2),
                                           len(edge_processed_image) - (filter_size - 2),
                                           len(edge_processed_image)
                                           )

    edge_processed_image = assign_to_range(edge_processed_image,
                                           select_range_2d(image, 0, len(image), len(image[0]) - (filter_size - 2),
                                                           len(image[0])),
                                           0, (filter_size - 2),
                                           (filter_size - 2), len(edge_processed_image) - (filter_size - 2)
                                           )

    edge_processed_image = assign_to_range(edge_processed_image,
                                           select_range_2d(image, 0, len(image), 0, (filter_size - 2)),
                                           len(edge_processed_image[0]) - (filter_size - 2),
                                           len(edge_processed_image[0]),
                                           (filter_size - 2), len(edge_processed_image) - (filter_size - 2), )
    return edge_processed_image


def edge_operation(image, filter_size):
    '''
    image wrapping from the edge pixel
    :param image: image as a list
    :return: edge processed image as a list
    '''
    # create a zero filled array for save the output
    edge_processed_image = generate_zero_array(len(image) + (filter_size - 2) * 2,
                                               len(image[0]) + (filter_size - 2) * 2, len(image[0][0]))

    for i in range(len(image[0][0])):
        # doing edge operation for one colour channel
        edged_2d = edge_operation_2d(select_range_3d(image, 0, len(image), 0, len(image[0]), i, i + 1), filter_size)

        # assign to the output
        for y in range(len(edge_processed_image)):
            for x in range(len(edge_processed_image[0])):
                edge_processed_image[y][x][i] = edged_2d[y][x]

    return edge_processed_image


def image_write(image, file_name):
    '''
    write image
    :param image: image as a list
    :param file_name: file name that want to save
    :return:
    '''
    cv2.imwrite(file_name + '.jpg', np.array(image))


def mean_filter_operation(image, filter_size=3):
    '''
    do the mean filtering operation
    :param image: image as a list
    :param filter_size: kernel size, default value is 3
    :return: mean filtered image as a list
    '''

    # generate the kernel
    kernel = generate_one_array(filter_size, filter_size)
    kernel = [[y / (filter_size ** 2) for y in x] for x in kernel]
    # since kernel is symmetric no need to flip

    # get dimension of the image
    x_shape = len(image[0])  # shape[1]
    y_shape = len(image)

    # generate zero filled list to save output
    output = generate_zero_array(len(image), len(image[0]), len(image[0][0]))

    # do edge operation
    edge_processed_img = edge_operation(img, filter_size)

    for i in range(len(edge_processed_img[0][0])):
        for x in range(x_shape):
            for y in range(y_shape):
                # conv operation
                output[y][x][i] = (conv_operation_sum(kernel, select_range_3d(edge_processed_img, y, y + filter_size, x,
                                                                              x + filter_size, i, i + 1)))

    return output


def median_filter_operation(image, filter_size=3):
    '''
       do the median filtering operation
       :param image: image as a list
       :param filter_size: kernel size, default value is 3
       :return: median filtered image as a list
    '''
    # generate the kernel
    kernel = generate_one_array(filter_size, filter_size)
    # since kernel is symmetric no need to flip

    # get dimension of the image
    x_shape = len(image[0])  # shape[1]
    y_shape = len(image)

    # generate zero filled list to save output
    output = generate_zero_array(len(image), len(image[0]), len(image[0][0]))

    # do edge operation
    edge_processed_img = edge_operation(img, filter_size)

    for i in range(len(edge_processed_img[0][0])):
        for x in range(x_shape):
            for y in range(y_shape):
                # apply median filter
                output[y][x][i] = conv_operation_median(kernel,
                                                        select_range_3d(edge_processed_img, y, y + filter_size, x,
                                                                        x + filter_size, i, i + 1))
    return output


def mid_filter_operation(image, filter_size=3):
    '''
       do the mid filtering operation
       :param image: image as a list
       :param filter_size: kernel size, default value is 3
       :return: mid filtered image as a list
    '''
    # generate the kernel
    kernel = generate_one_array(filter_size, filter_size)
    # since kernel is symmetric no need to flip

    # get dimension of the image
    x_shape = len(image[0])  # shape[1]
    y_shape = len(image)

    # generate zero filled list to save output
    output = generate_zero_array(len(image), len(image[0]), len(image[0][0]))

    # do edge operation
    edge_processed_img = edge_operation(img, filter_size)

    for i in range(len(edge_processed_img[0][0])):
        for x in range(x_shape):
            for y in range(y_shape):
                # apply mid filter
                output[y][x][i] = (conv_operation_mid(kernel, select_range_3d(edge_processed_img, y, y + filter_size, x,
                                                                              x + filter_size, i, i + 1)))
    return output


def generate_zero_array(x, y, z):
    '''
    generate 3d list with zeros
    :param x: x dimension
    :param y: y dimension
    :param z: z dimension
    :return: 3d list of zeros
    '''
    return [[[0 for k in range(z)] for j in range(y)] for i in range(x)]


def generate_one_array(x, y):
    '''
    generate 2d list with ones
    :param x: x dimension
    :param y: y dimension
    :return: 2d list of ones
    '''
    return [[1 for j in range(y)] for i in range(x)]


def generate_zero_array_2d(x, y):
    '''
    generate 2d list with zeros
    :param x: x dimension
    :param y: y dimension
    :return: 2d list of zeros
    '''
    return [[0 for j in range(y)] for i in range(x)]


def select_range_2d(arr, x1, x2, y1, y2):
    '''
    select a range of a given 2d list
    :param arr: 2d list
    :param x1: x from
    :param x2: x to
    :param y1: y from
    :param y2: y to
    :return: 2d list of selected range
    '''
    return [i[y1:y2] for i in arr[x1:x2]]


def select_range_3d(arr, x1, x2, y1, y2, z1, z2):
    '''
    select a range of a given 3d list
    :param arr: 3d list
    :param x1: x from
    :param x2: x to
    :param y1: y from
    :param y2: y to
    :param z1: z from
    :param z2: z to
    :return:
    '''
    selected_range = []
    for i in arr[x1:x2]:
        temp = []
        for j in i[y1:y2]:
            for k in j[z1:z2]:
                temp.append(k)
        selected_range.append(temp)
    return selected_range


def assign_to_range(arr, assigner, x1, x2, y1, y2):
    '''
    assign the given 2d list for given range
    :param arr: 2d list
    :param assigner: 2d list
    :param x1: x from
    :param x2: x to
    :param y1: y from
    :param y2: y to
    :return: 2d list: assigner is assigned in arr[x1:x2][y1:y2]
    '''
    x, y, z = 0, 0, 0
    for i in range(x1, x2):
        y = 0
        for j in range(y1, y2):
            arr[j][i] = assigner[y][x]
            y += 1
        x += 1
    return arr


def conv_operation_sum(x, y):
    '''
    get sum of the convulutional operation
    :param x: kernel
    :param y: selected window
    :return: sum of the operation
    '''
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i][j] * y[i][j]
    return result


def conv_operation_median(x, y):
    '''
    median of the convolutional operation
    :param x: kernel
    :param y: selected window
    :return: median of the operation
    '''
    return median(conv_1d_lst(x, y))


def conv_operation_mid(x, y):
    '''
    mid of the convolutional operation
    :param x: kernel
    :param y: selected window
    :return: mid of the operation
    '''
    return mid(conv_1d_lst(x, y))


def conv_1d_lst(x, y):
    '''
    1d array the convolutional operation
    :param x: kernel
    :param y: selected window
    :return: 1d array of the convolutional operation
    '''
    result = []
    for i in range(len(x)):
        for j in range(len(y)):
            result.append(x[i][j] * y[i][j])
    return result


def median(lst):
    '''
    median of a list
    :param lst: list
    :return: median of the list
    '''
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2

    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1]) / 2.0


def mid(lst):
    '''
    mid of a list
    :param lst: list
    :return: mid of the list
    '''
    sortedLst = sorted(lst)
    # sortedLst[0] = minimum value
    # sortedLst[1] = maximum value
    return (sortedLst[0] + sortedLst[-1]) / 2


if __name__ == '__main__':

    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        splited = f.split('.')

        if splited[1] == 'jpg' or splited[1] == 'jpeg' or splited[1] == 'JPEG' or splited[1] == 'JPG':
            img = read_image(f)

            image_write(mean_filter_operation(img, 3), splited[0] + '_mean_filter_output')
            image_write(median_filter_operation(img, 3), splited[0] + '_median_filter_output')
            image_write(mid_filter_operation(img, 3), splited[0] + '_mid_filter_output')
