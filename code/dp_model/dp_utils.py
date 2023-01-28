import numpy as np
from scipy.stats import norm

def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        


def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example:
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    if min(np.array(in_sp)-np.array(out_sp)) < 0:
        data_large_shape = []
        for i in range(0, 3):
            data_large_shape.append(max(in_sp[i],out_sp[i]))
        data_large_shape = tuple(data_large_shape)
        #print('large:', data_large_shape)
        data_large = np.zeros(data_large_shape)

        x_crop = int(abs(in_sp[-3] - data_large_shape[-3]) / 2)
        y_crop = int(abs(in_sp[-2] - data_large_shape[-2]) / 2)
        z_crop = int(abs(in_sp[-1] - data_large_shape[-1]) / 2)
        data_large[x_crop:x_crop+in_sp[-3], y_crop:y_crop+in_sp[-2], z_crop:z_crop+in_sp[-1]] = data
        data = data_large
    nd = np.ndim(data)
    #print(x_crop, y_crop, z_crop)
    #print(x_addone, y_addone, z_addone)
    #x_crop, y_crop, z_crop = abs(x_crop), abs(y_crop), abs(z_crop)
    in_sp = data.shape
    x_crop = int(abs(in_sp[-3] - out_sp[-3]) / 2)
    y_crop = int(abs(in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int(abs(in_sp[-1] - out_sp[-1]) / 2)
    data_crop = data[x_crop:x_crop+out_sp[-3], y_crop:y_crop+out_sp[-2], z_crop:z_crop+out_sp[-1]]
    return data_crop



def crop_center_v1(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example:
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    x_addone = in_sp[-3]%2
    y_addone = in_sp[-2]%2
    z_addone = in_sp[-1]%2
    #print(x_crop, y_crop, z_crop)
    #print(x_addone, y_addone, z_addone)
    if nd == 3:
        data_crop = data[x_crop:-x_crop-x_addone, y_crop:-y_crop-y_addone, z_crop:-z_crop-z_addone]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop-x_addone, y_crop:-y_crop-y_addone, z_crop:-z_crop-z_addone]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

def crop_center_origin(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x_1 = 60.5
    bin_range = [42, 82]
    bin_step = 1
    sigma = 1
    vect_1 = num2vect(x_1, bin_range, bin_step, sigma)
    print(vect_1)
    plt.bar(vect_1[1], vect_1[0], color='b', alpha = 0.2, label = '60 years old')


    x_2 = 42.3
    vect_2 = num2vect(x_2, bin_range, bin_step, sigma)
    print(vect_2)
    plt.bar(vect_2[1], vect_2[0], color='g', alpha=0.2, label='42 years old')

    x_3 = 82.5
    vect_3 = num2vect(x_3, bin_range, bin_step, sigma)
    print(vect_3)
    plt.bar(vect_3[1], vect_3[0], color='r', alpha=0.2, label='82 years old')
    plt.title('Probability generated by normal distribution (sigam=1)')

    plt.legend()
    plt.show()