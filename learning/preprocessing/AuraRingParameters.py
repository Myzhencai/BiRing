import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
import pickle
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from mag_utils import DeviceCalibration, compute_sensor, get_field, get_field_new,compute_sensor_new,plot_correlation, get_sensor_pos


CROSSTALK_MATRIX = False
CROSS_BIAS = False
POLY_FIT = False
POLY_ORDER = 11
SENSOR_ROT = False

# 需要学习的参数
GLOBAL_GAIN =True
GAIN =True
PER_CHANNEL_GAIN =True
BIAS  =True
NOISE =True
SENSOR_OFFSET =True
SENSOR_ROT_OFFSET =True
RING_OFFSET =True
RING_ROT_OFFSET =True
CROSSTALK =True
BASE_SENSOR_OFFSET =True
BASE_SENSOR_ROT_OFFSET =True
USE_INTERP_CORRECTION = True
USE_SIGN_CHANGE = False


def measure_error(calibrated, calibrated_samples):
    mag = np.linalg.norm(calibrated, axis=1)
    error = mag - 1
    error = error[~np.isnan(error)]
    error_mag = np.mean(error**2)*2
    # dot_prod = np.dot(calibrated, [-1,0,0])
    # error = dot_prod - 1
    # error_mag = np.mean(error ** 2) * 2

    if USE_SIGN_CHANGE:
        samples = [sample[:, dim] for sample, dim in calibrated_samples]
        samples = np.nan_to_num(samples)
        x = np.linspace(0, 1, len(samples[0]))
        rs = [scipy.stats.pearsonr(x, y)[0] for y in samples]
        rs = np.nan_to_num(rs)
        # print(matrix, bias, np.mean(error**2))
        error_sign = np.mean((np.array(rs)**2-1)**2)*5
    else:
        error_sign = 0

    return error_mag + error_sign


def filter_bad_data(data, use_abs=False):
    print(data.shape)

    if use_abs:
        transform = lambda x: np.abs(x)
    else:
        transform = lambda x: x

    filtered = scipy.signal.savgol_filter(transform(data), 15, 2, axis=0)
    error = np.sqrt(np.sum((transform(data)-filtered)**2, axis=1))
    # plt.figure()
    # plt.plot(data)
    # plt.figure()
    # plt.plot(error)
    # plt.show()
    return data[error < .007, :]


# data 相当于x
def decode_x(data):
    i = 0
    global_gain = data[i]
    i += 1
    gain = data[i:i + 3]
    i += 3
    per_channel_gain = data[i:i + 3]
    i += 3
    bias = data[i:i + 3]
    i += 3
    noise = data[i:i + 3]
    i += 3
    sensor_offset = data[i:i + 3]
    i += 3
    sensor_rot_offset = R.from_quat([data[i + 1], data[i + 2], data[i + 3], data[i]])  # x, y, z, w
    i += 4
    ring_offset = data[i:i + 3]
    i += 3
    ring_rot_offset = R.from_quat([data[i + 1], data[i + 2], data[i + 3], data[i]])  # x, y, z, w
    i += 4
    crosstalk = data[i:i + 3]
    i += 3
    base_sensor_offset = data[i:i + 3]
    i += 3
    base_sensor_rot_offset = R.from_quat([data[i + 1], data[i + 2], data[i + 3], data[i]])  # x, y, z, w
    i += 4
    return global_gain,gain,per_channel_gain,bias,noise,sensor_offset,sensor_rot_offset,\
        ring_offset,ring_rot_offset,crosstalk,base_sensor_offset,base_sensor_rot_offset


def encode_x(global_gain,gain,per_channel_gain,bias,noise,sensor_offset,sensor_rot_offset,
                  ring_offset,ring_rot_offset,crosstalk,base_sensor_offset,base_sensor_rot_offset):
    x = []
    if GLOBAL_GAIN:
        x +=global_gain
    if  GAIN:
        x +=gain
    if PER_CHANNEL_GAIN:
        x += per_channel_gain
    if BIAS:
        x += bias
    if NOISE:
        x += noise
    if SENSOR_OFFSET:
        x += sensor_offset
    if SENSOR_ROT_OFFSET:
        x += sensor_rot_offset
    if RING_OFFSET:
        x += ring_offset
    if RING_ROT_OFFSET:
        x += ring_rot_offset
    if CROSSTALK:
        x += crosstalk
    if BASE_SENSOR_OFFSET:
        x +=base_sensor_offset
    if BASE_SENSOR_ROT_OFFSET:
        x +=base_sensor_rot_offset
    return np.array(x)



def get_bounds():
    bounds = []
    if PER_CHANNEL_GAIN:
        bounds += [(.5,5)] * 3

    if CROSSTALK_MATRIX:
        for i in range(9):
            if i % 4 == 0:
                bounds += [(.999, 1.001)]
            else:
                bounds += [(-.5, 0)]

    if CROSS_BIAS:
        bounds += [(-.2, 0)] * 3
    if BIAS:
        bounds += [(-.1, .1)] * 3
    if POLY_FIT:
        bounds += [(-20, 20)] * (POLY_ORDER-1) + [(-7e-03, 7e-03)]
    if SENSOR_ROT:
        bounds += [(-1, 1)] * 4

    return bounds


def find_zero_crossings(data):
    samples = []
    plt.figure()
    for (i, dim) in zip(*np.where(np.diff(np.sign(data), axis=0))):
        diff = data[i + 1, dim] - data[i, dim]
        if diff < 0.1:
            print(diff)
            sample = data[i-500:i+500, :]
            plt.plot(sample[:,dim])
            samples.append((sample, dim))
    plt.show()
    return samples


def fix_signs(data):
    # plt.plot(data)
    for (i, dim) in zip(*np.where(np.diff(np.sign(data), axis=0))):
        no_switch_diff = data[i + 1, :] - data[i, :]
        switch_diff = -data[i + 1, :] - data[i, :]
        if np.linalg.norm(no_switch_diff) > np.linalg.norm(switch_diff):
            data[i+1:, :] *= -1
    # plt.figure()
    # plt.plot(data)
    # plt.show()
    return data


def filter_and_downsample(x, factor):
    # b, a = scipy.signal.butter(8, 40 / (3960/2), btype='lowpass')
    # x_filt = scipy.signal.filtfilt(b, a, x, axis=0)
    x_filt = x
    # plt.plot(x)
    # plt.plot(x_filt)
    # plt.show()
    # return scipy.signal.decimate(x_filt, factor)
    return x_filt[::factor]

def load_resampled_data_new(path):
    all_data = np.loadtxt(path)
    mag_data = all_data[:, 6:9]
    # mag_data = mag_data[:, [0, 2, 1]]#可能不需要调整
    pos = all_data[:, 0:3]*0.001
    rot = all_data[:, 3:6]
    return mag_data, pos, rot


# ma 工的数据中是瞬间的读数还是滤波和翻转求了平均值的数据？
def main():
    # 加载实际的Sensor的读数以及对应的Pose和Rotation数据
    # dataPath = "/home/gaofei/BiRing/Data/processed__t1.txt"
    dataPath = "/home/gaofei/Aura/Aruadata/test_3traindata.txt"
    needchange = True
    # needchange = True
    mag_data, pos, rot = load_resampled_data_new(dataPath)

    # 筛选实际的额数据（可借鉴Aura的方法）filterBad
    print("filterBad")

    # 欧拉角转4元数
    if needchange:
        # theta = [-116, 0., -105]
        # r6 = R.from_euler('xyz', theta, degrees=True)
        # rm = r6.as_quat()
        # 已经确定是
        rot = [R.from_euler('xyz', theta, degrees=True).as_quat() for theta in rot]
        # rot = [Quaternion(x) for x in rot]
    if USE_SIGN_CHANGE:
        samples = find_zero_crossings(mag_data)
    else:
        samples = []



    # 需要优化的目标更具BiRing中的计算方法
    def costBiRing(x):
        # 将封装的需要优化的参数先解压出来
        global_gain, gain, per_channel_gain, bias, noise, sensor_offset, sensor_rot_offset, \
            ring_offset, ring_rot_offset, crosstalk, base_sensor_offset, base_sensor_rot_offset = decode_x(x)
        # 用解压的参数参与模型公式计算
        field =get_field_new(pos, rot, base_sensor_offset,sensor_offset,sensor_rot_offset,base_sensor_rot_offset,ring_offset,ring_rot_offset)
        calibrated = compute_sensor_new(field, global_gain, per_channel_gain, 0, 0, 0, noise, gain, bias)
        # 计算真实数据和预测数据的误差
        # calibrated_samples = [(compute_sensor(get_field(pos, rot, x), x), dim) for sample, dim in samples]
        # error = measure_error(calibrated, calibrated_samples)
        error_train = calibrated - mag_data
        # 可以换成其它算法
        # error = np.mean(np.sqrt(np.sum(np.square(error_train), axis=1)))
        error = np.sum(np.sqrt(np.sum(np.square(error_train), axis=1)))
        print("current error", error)
        return error

    # 初始化需要优化的参数
    global_gain = [2]
    gain = [7,9,13]
    per_channel_gain = [5,5,5]
    bias = [0.8,0.6,0.2]
    noise =[0.4,0.2,0.6]
    sensor_offset = [0.2,0.3,0.4]
    sensor_rot_offset =[1,0.8,0.6,0]  # x, y, z, w
    ring_offset = [0.1,0.3,0.9]
    ring_rot_offset =[1,0.3,0.7,0.2]  # x, y, z, w
    crosstalk = [0.2,0.9,0.1]
    base_sensor_offset = [0.2,0.4,0.5]
    base_sensor_rot_offset = [1,0.3,0.9,0]

    # 将需要优化的参数打包
    x0 = encode_x(global_gain,gain,per_channel_gain,bias,noise,sensor_offset,sensor_rot_offset,
                  ring_offset,ring_rot_offset,crosstalk,base_sensor_offset,base_sensor_rot_offset)
    optionsmap = {"maxiter": 80000,'disp': True}
    print("Running minimizer...")
    # x_opt = scipy.optimize.minimize(costBiRing, x0=x0, bounds=get_bounds())

    x_opt = scipy.optimize.minimize(costBiRing, x0=x0,options=optionsmap)
    print("final result is :",x_opt.x)

    #将学习到的参数解码
    # calibrated = compute_sensor(get_field(pos, rot, x), x)
    # compute_sensor_new(field, global_gain, per_channel_gain, kxy, kxz, kyz, noise, gain, bias)

    global_gain_x, gain_x, per_channel_gain_x, bias_x, noise_x, sensor_offset_x, sensor_rot_offset_x, \
        ring_offset_x, ring_rot_offset_x, crosstalk_x, base_sensor_offset_x, base_sensor_rot_offset_x = decode_x(x_opt.x)
    print("global_gain:{0}\n gain{1}\n  per_channel_gain{2}\n  bias{3}\n  noise{4}\n  sensor_offset{5}\n  sensor_rot_offset{6}\n  \
        ring_offset{7}\n  ring_rot_offset{8}\n  crosstalk{9}\n  base_sensor_offset{10}\n  base_sensor_rot_offset{11}\n".
          format(global_gain_x, gain_x, per_channel_gain_x, bias_x, noise_x, sensor_offset_x, sensor_rot_offset_x, \
        ring_offset_x, ring_rot_offset_x, crosstalk_x, base_sensor_offset_x, base_sensor_rot_offset_x))

    calibrated = compute_sensor_new(
        get_field_new(pos, rot, base_sensor_offset_x, sensor_offset_x, sensor_rot_offset_x, base_sensor_rot_offset_x,
                      ring_offset_x, ring_rot_offset_x), global_gain_x, per_channel_gain_x, 0, 0, 0, noise_x, gain_x, bias_x)

    print("calibrated :",calibrated)
    print("real mag :",mag_data)




    # calibrated = costBiRing(mag_data, x_opt.x, rot, apply_rot=False)
    # calibrated_rot = apply_calibration_x(mag_data, x_opt.x, rot, apply_rot=True)

    # per_channel_gain = [1.73543888, 1.73543888, 1.73543888]
    # bias = [0, 0, 0]
    # crosstalk = np.eye(3)
    # crosstalk[0,1] = -5e-4
    # crosstalk[2,1] = -5e-4
    # cross_bias = [1, 1, 1]
    # x0 = encode_x(per_channel_gain, crosstalk, cross_bias, bias)
    # calibrated2 = apply_calibration_x(data, x0)
    # measure_error(data, [(apply_calibration_x(sample, [0, 0, 0], rot_quat), dim) for sample, dim in samples])
    # plt.figure()
    # plt.title("Original data (after interpolation)")
    # plt.plot(data)
    # plt.figure()
    # plt.title("Calibrated data")
    # plt.plot(calibrated)
    # plt.figure()
    # plt.title("Abs Calibrated data")
    # plt.plot(np.abs(calibrated))
    # plt.figure()
    # plt.title("Calibrated data after rotation")
    # plt.plot(calibrated_rot)
    # plt.figure()
    # plt.plot(calibrated2)
    #
    # plt.figure()
    # plt.title("Magnitudes")
    # plt.plot(np.linalg.norm(data, axis=1))
    # plt.plot(np.linalg.norm(calibrated, axis=1))
    # plt.show()

if __name__ == "__main__":
    main()
