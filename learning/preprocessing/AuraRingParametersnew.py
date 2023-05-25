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


# 解码一维的需要训练的参数
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
    # 此处可能有问题
    sensor_rot_offset = R.from_quat([data[i + 1], data[i + 2], data[i + 3], data[i]])  # x, y, z, w
    debugvalue = sensor_rot_offset.as_quat()
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


# 编码一维的需要训练的参数
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


# 设置每个参数的学习范围以及不等式条件式等
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

def get_bounds_new():
    bounds = []
    if GLOBAL_GAIN:
        bounds += [(0, 100)]
    if GAIN:
        bounds += [(0, 10000)]*3
    if PER_CHANNEL_GAIN:
        bounds += [(0, 10000)] * 3
    if BIAS:
        bounds += [(0, 0.2)] * 3
    if NOISE:
        bounds += [(0, 3)] * 3
    if SENSOR_OFFSET:
        bounds += [(-10, 10)] * 3
    if SENSOR_ROT_OFFSET:
        bounds += [(-1, 1)] * 4
    if RING_OFFSET:
        bounds += [(-10, 10)] * 3
    if RING_ROT_OFFSET:
        bounds += [(-1, 1)] * 4
    if CROSSTALK:
        bounds += [(-5, 5)] * 3
    # 自己定义的范围
    if BASE_SENSOR_OFFSET:
        bounds += [(-5, 5)] * 3
    if BASE_SENSOR_ROT_OFFSET:
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
    pos = all_data[:, 0:3]*0.0001
    rot = all_data[:, 3:6]
    return mag_data, pos, rot

def dipole_model_newly(pos):
    # https://ccmc.gsfc.nasa.gov/RoR_WWW/presentations/Dipole.pdf
    pos = np.atleast_2d(pos)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    Bx = 3*x*z/(r**5)
    By = 3*y*z/(r**5)
    Bz = (3*z**2-r**2)/(r**5)
    field = np.vstack((Bx, By, Bz)).T

    return field * 1e5

def get_field_newly(ring_pos, ring_q, base_sensor_offset, sensor_offset, sensor_rot_offset, base_sensor_rot_offset,
                  ring_offset, ring_rot_offset):
    # 直接通过空间位置和旋转来得到理论的场强值
    # 此处的顺序可能需要调整？旋转需要
    ring_q = np.atleast_2d(ring_q)[:, [1, 2, 3, 0]]
    sensor = base_sensor_offset + sensor_offset
    sensor_rot = sensor_rot_offset * base_sensor_rot_offset

    ring_qs = R.from_quat(ring_q)
    ring_pos_adj = ring_pos + ring_qs.apply(ring_offset, inverse=True)
    ring_rot_adj = ring_qs * ring_rot_offset

    sensor_ring = (ring_qs * ring_rot_offset).apply(sensor - ring_pos_adj)
    field = ring_rot_adj.apply(dipole_model_newly(sensor_ring), inverse=True)

    return sensor_rot.apply(field)

def compute_sensor_newly(pos, rot, x):
    # 解析出当前的需要优化的参数
    global_gain, gain, per_channel_gain, bias, noise, sensor_offset, sensor_rot_offset, \
        ring_offset, ring_rot_offset, crosstalk, base_sensor_offset, base_sensor_rot_offset = decode_x(x)
    kxy,kxz,kyz,kxz =0,0,0,0

    # 计算场强值
    # ring_q = rot
    # 可能不需要进行这个处理
    # ring_q = np.atleast_2d(rot)[:, [1, 2, 3, 0]]
    ring_q =rot
    # 在这里有个错误需要排查原因
    sensor = base_sensor_offset + sensor_offset
    sensor_rot = R.from_quat([-0.25179998703002937,-0.81240000078678065,0.60779995710849732,-0.64770000081062307])
    # sensor_rot = sensor_rot_offset * base_sensor_rot_offset
    # debugvalue = sensor_rot.as_quat()
    # senssor_rot = R.from_quat([-0.520000, 0.530000, 0.560000, -0.520000])*R.from_quat([0,0.3,0.9,1])
    # debugvalue2 = senssor_rot.as_quat()
    # print(sensor_rot.as_quat())
    ring_qs = R.from_quat(ring_q)
    ring_pos_adj = pos + ring_qs.apply(ring_offset, inverse=True)
    ring_rot_adj = ring_qs * ring_rot_offset

    sensor_ring = (ring_qs * ring_rot_offset).apply(sensor - ring_pos_adj)
    field = ring_rot_adj.apply(dipole_model_newly(sensor_ring), inverse=True)
    field = sensor_rot.apply(field)

    # 计算根据公式计算理论的预测读数值
    field_adj = field * global_gain / per_channel_gain
    # field_adj = np.abs(field * global_gain / per_channel_gain)
    coeffs = np.array([[1, kxy ** 2, kxz ** 2],
                       [kxy ** 2, 1, kyz ** 2],
                       [kxz ** 2, kyz ** 2, 1],
                       [2 * kxy * kxz, 0, 0],
                       [0, 2 * kxy * kyz, 0],
                       [0, 0, 2 * kxz * kyz],
                       noise ** 2])

    features_1 = np.hstack((field_adj ** 2,
                            np.array([field_adj[:, 1] * field_adj[:, 2],
                                      field_adj[:, 0] * field_adj[:, 2],
                                      field_adj[:, 0] * field_adj[:, 1]]).T, np.ones((field_adj.shape[0], 1))))
    sensors = gain * np.sqrt(np.matmul(features_1, coeffs)) - bias

    sensors[sensors < 0] = 0

    return sensors


def main():
    # 加载实际的Sensor的读数以及对应的Pose和Rotation数据
    dataPath = "/home/gaofei/Aura/Aruadata/reciver_to_sender_6dofwithmag.txt"
    needchange = True
    mag_data, pos, rot = load_resampled_data_new(dataPath)

    # 筛选实际的额数据（可借鉴Aura的方法）filterBad
    print("filterBad")
    # mag_data = filter_bad_data(mag_data)
    # 转为4元数
    if needchange:
         # rot = [Quaternion(x) for x in rot]
         rot = [R.from_euler('xyz', theta, degrees=True).as_quat() for theta in rot]
    if USE_SIGN_CHANGE:
        samples = find_zero_crossings(mag_data)
    else:
        samples = []

    # 需要优化的目标更具BiRing中的计算方法
    def costBiRing(x):
        # 计算出在固定空间位置上符合我们自己设定的方程的预测值
        calibrated_new = compute_sensor_newly(pos, rot, x)
        # 计算真实数据和预测数据的误差
        # 如果强度太低就不参与运算？
        error_train = calibrated_new - mag_data
        # 可以换成其它算法
        # error = np.mean(np.sqrt(np.sum(np.square(error_train), axis=1)))
        error = np.sum(np.sqrt(np.sum(np.square(error_train), axis=1)))
        print("current error", error)
        return error

    # 初始化需要优化的参数
    global_gain = [20]
    gain = [700,900,908]
    per_channel_gain = [500,500,500]
    bias = [0.1,0.1,0.1]
    noise =[2,2,2]
    sensor_offset = [-4,8,8]
    sensor_rot_offset =[1,0.8,0.6,0]  # x, y, z, w
    ring_offset = [8,9,9]
    ring_rot_offset =[1,0.3,0.7,0.2]  # x, y, z, w
    crosstalk = [3,2,4]
    base_sensor_offset = [0.2,0.4,0.5]
    base_sensor_rot_offset = [1,0.3,0.9,0]


    # 将需要优化的参数打包
    x0 = encode_x(global_gain,gain,per_channel_gain,bias,noise,sensor_offset,sensor_rot_offset,
                  ring_offset,ring_rot_offset,crosstalk,base_sensor_offset,base_sensor_rot_offset)
    optionsmap = {"maxiter": 1050,'disp': True}
    print("Running minimizer...")
    # x_opt = scipy.optimize.minimize(costBiRing, x0=x0, bounds=get_bounds())

    # x_opt = scipy.optimize.minimize(costBiRing, x0=x0,options=optionsmap,bounds=get_bounds_new())
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

    # calibrated = costBiRing(x_opt.x)
    calibrated_new = compute_sensor_newly(pos, rot, x_opt.x)

    # caculate_init = compute_sensor_newly(pos, rot, x0)
    #
    # print(caculate_init)

    print("calibrated :",calibrated_new)
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
#
def creat3d_vec_xyzold(start_points, end_points, colors=None, view_angle=20, azimuth=30):
    '''
    创建一个空间三维坐标系
    :param start_points: 绘制的数据起点
    :param end_points: 绘制的数据终点
    :param colors: 绘制的向量颜色
    :param view_angle: 观察角度
    :param azimuth: 方位角
    :return:
    '''
    assert start_points.shape == end_points.shape

    if colors is None:
        colors = np.random.randint(0, 255, size=start_points.shape, dtype=np.uint8)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = fig.gca(projection='3d')
    num_vec = start_points.shape[0]

    # q = ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2], end_points[:, 0], end_points[:, 1],
    #               end_points[:, 2], color="#666666", arrow_length_ratio=0.1)
    for i in range(num_vec):
        color = '#'
        for j in range(3):
            color += str(hex(colors[i, j]))[-2:].replace('x', '0').upper()
        q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points[i, 0], end_points[i, 1],
                      end_points[i, 2], color=color, arrow_length_ratio=0.1)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    ax.view_init(view_angle, azimuth)
    ax.set_title('coordinates-xyz')
    plt.show()

def creat3d_vec_xyz(start_points, end_points, end_points1,colors=None, view_angle=20, azimuth=30):
    '''
    创建一个空间三维坐标系
    :param start_points: 绘制的数据起点
    :param end_points: 绘制的数据终点
    :param colors: 绘制的向量颜色
    :param view_angle: 观察角度
    :param azimuth: 方位角
    :return:
    '''
    assert start_points.shape == end_points.shape

    if colors is None:
        colors = np.random.randint(0, 255, size=start_points.shape, dtype=np.uint8)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = fig.gca(projection='3d')
    num_vec = start_points.shape[0]

    # q = ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2], end_points[:, 0], end_points[:, 1],
    #               end_points[:, 2], color="#666666", arrow_length_ratio=0.1)
    for i in range(num_vec):
        color = '#'
        for j in range(3):
            color += str(hex(colors[i, j]))[-2:].replace('x', '0').upper()
        q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points[i, 0], end_points[i, 1],
                      end_points[i, 2], color=color, arrow_length_ratio=0.1)
    for i in range(num_vec):
        color = '#'
        for j in range(3):
            color += str(hex(colors[i, j]))[-2:].replace('x', '0').upper()
        q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points1[i, 0], end_points1[i, 1],
                      end_points1[i, 2], color=color, arrow_length_ratio=0.1)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    ax.view_init(view_angle, azimuth)
    ax.set_title('coordinates-xyz')
    plt.show()


# def creat3d_vec_xyz(start_points, end_points, end_points1, end_points2, colors=None, view_angle=20, azimuth=30):
#     '''
#     创建一个空间三维坐标系
#     :param start_points: 绘制的数据起点
#     :param end_points: 绘制的数据终点
#     :param colors: 绘制的向量颜色
#     :param view_angle: 观察角度
#     :param azimuth: 方位角
#     :return:
#     '''
#     assert start_points.shape == end_points.shape
#
#     if colors is None:
#         colors = np.random.randint(0, 255, size=start_points.shape, dtype=np.uint8)
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     # ax = fig.gca(projection='3d')
#     num_vec = start_points.shape[0]
#
#     # q = ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2], end_points[:, 0], end_points[:, 1],
#     #               end_points[:, 2], color="#666666", arrow_length_ratio=0.1)
#     for i in range(num_vec):
#         color = '#'
#         for j in range(3):
#             color += str(hex(colors[i, j]+5))[-2:].replace('x', '0').upper()
#         q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points[i, 0], end_points[i, 1],
#                       end_points[i, 2], color=color, arrow_length_ratio=0.1)
#
#     for i in range(num_vec):
#         color = '#'
#         for j in range(3):
#             color += str(hex(colors[i, j]))[-2:].replace('x', '0').upper()
#         q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points1[i, 0], end_points1[i, 1],
#                       end_points1[i, 2], color=color, arrow_length_ratio=0.1)
#
#     for i in range(num_vec):
#         color = '#'
#         for j in range(3):
#             color += str(hex(colors[i, j]-5))[-2:].replace('x', '0').upper()
#         q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points2[i, 0], end_points2[i, 1],
#                       end_points2[i, 2], color=color, arrow_length_ratio=0.1)
#
#     # ax.set_xlim(0, 1)
#     # ax.set_ylim(0, 1)
#     # ax.set_zlim(0, 1)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
#     ax.view_init(view_angle, azimuth)
#     ax.set_title('coordinates-xyz')
#     plt.show()

if __name__ == "__main__":
    '''
    灯会子验证
    '''
    base = np.eye(3)
    newbase = np.c_[base,np.array([1,1,1])]
    # base0 = np.eye(3)[:,0]
    # base1 = np.eye(3)[:,1]
    # base2 = np.eye(3)[:,2]
    print(base)
    a2 = R.from_euler('xyz', [90, -80, 50], degrees=True).as_matrix()
    # a3 = R.from_euler('xyz', [90, -80, 50], degrees=True).as_quat()

    # a4 = R.from_quat(a3).as_matrix()
    a1 = R.from_euler('xyz', [90, 0, 50], degrees=True).as_matrix()
    x = a1.dot(a2)
    # x = R.from_matrix(x).as_quat()
    x = np.c_[x,np.array([3,3,3])]
    x = np.r_[x,np.array([0,0,0,1]).reshape(1,4)]

    newresult = x.dot(newbase.T)
    print(x)
    rotated =a1.dot(a2.dot(base))

    # rotatedx = a2*(base0.T)
    # rotatedy = a2*(base1.T)
    # rotatedz = a2*(base2.T)
    #
    # resultrotated = np.c_[rotatedx,rotatedy]
    # resultrotated = np.c_[resultrotated,rotatedz]
    # print(rotated)

    start_points = np.zeros((3, 3))
    test = newresult[:3]
    basenew = np.array([[3,3,3],[3,3,3],[3,3,3]])
    # creat3d_vec_xyz(start_points, basenew,test)
    creat3d_vec_xyzold(basenew,test.T)
    # creat3d_vec_xyz(start_points, rotated)

    # plt.figure()
    # # plt.plot(mag_data,c="r")
    # ax1 = plt.axes(projection='3d')
    # # ax1.scatter3D(mag_data[:,0], mag_data[:,1], mag_data[:,2], c='red')
    # # ax1.scatter3D(mag_data[:,3], mag_data[:,4], mag_data[:,5], c='blue')
    # # ax1.scatter3D(mag_data[:,6], mag_data[:,7], mag_data[:,8], c='yellow')
    # ax1.scatter3D(rotated[:, 0], rotated[:, 1], rotated[:, 2], c='yellow')
    # # plt.plot(pos_train)
    # plt.title("TRAIN")
    # plt.show()

    a1 = R.from_euler('xyz', [90, -80, 50], degrees=True).as_quat()
    a2 = R.from_euler('xyz', [90, 0, 50], degrees=True).as_quat()

    result = a1*a2
    print(result)
    a3 = R.from_quat([result[1],result[2],result[3],result[0]]).as_matrix()
    rotatedvalue = a3.dot(base)
    creat3d_vec_xyz(start_points, rotated, rotatedvalue)
    # Ra =b
    Ra_b = rotatedvalue.dot(np.linalg.inv(rotated))
    angle = R.from_matrix(Ra_b).as_euler('xyz', degrees=True)
    print(angle)
    # a = R.from_quat([a1[0],a1[1],a1[2],a1[3]])
    # b = R.from_quat([a2[0],a2[1],a2[2],a2[3]])
    # c = a*b
    # d = c.as_quat()
    # e = a.as_quat()
    # # e2 = a2.as_quat()
    # aMatrix = R.from_euler('xyz', [90, 80, 30], degrees=True).as_matrix()
    # eigen = np.array([[-0.017027392411267384,0.109066087809429,0.99388865392337522],[-0.57976842786766758,0.80878283702102327,-0.098685827696169764],[-0.81460336235221009,-0.57790562467428097,0.049461611521494531]])
    # newm = np.linalg.inv(eigen)*aMatrix
    # print(d)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # dataPath = "/home/gaofei/Aura/Aruadata/reciver_to_sender_6dofwithmag.txt"
    # needchange = True
    # SHOW_FIGURE = True
    # mag_data, pos, rot = load_resampled_data_new(dataPath)
    #
    # if needchange:
    #     # rot = [Quaternion(x) for x in rot]
    #     rot = [R.from_euler('xyz', theta, degrees=True).as_quat() for theta in rot]

    # pos =np.array([-1.80129,0.118608,-0.850532])
    # rot =R.from_euler('xyz', [30,40,90], degrees=True).as_quat()
    # R.from_quat()
    # r = R.from_quat([0.256222, -0.907750999997, -1.82636, -1.837560])
    # rot =r.as_quat()
    ##############################
    # global_gain = [22]
    # gain = [320.000000, 240.000000, 400.000000]
    # per_channel_gain = [30.000000, 59.500000, 72.500000]
    # bias = [0.155000, 0.156000, 0.145500]
    # noise = [2.510000, 1.350000, 1.560000]
    # sensor_offset = [-5.500000, 6.500000, 4.520000]
    # sensor_rot_offset = [-0.520000, 0.530000, 0.560000, -0.520000] # x, y, z, w
    # ring_offset = [-8.500000, 4.500000, 7.500000]
    # ring_rot_offset = [-0.530000, 0.600000, -0.400000, 0.800000]  # x, y, z, w
    # crosstalk = [-0.4,0.36,-0.42]
    # base_sensor_offset = [8,8.7,6.8]
    # base_sensor_rot_offset = [0.49,0.86, -0.32, -0.5]
    #######################################
    # global_gain = [20]
    # gain = [700, 900, 908]
    # per_channel_gain = [500, 500, 500]
    # bias = [0.1, 0.1, 0.1]
    # noise = [2, 2, 2]
    # sensor_offset = [-4, 8, 8]
    # sensor_rot_offset = [1, 0.8, 0.6, 0]  # x, y, z, w
    # ring_offset = [8, 9, 9]
    # ring_rot_offset = [1, 0.3, 0.7, 0.2]  # x, y, z, w
    # crosstalk = [3, 2, 4]
    # base_sensor_offset = [0.2, 0.4, 0.5]
    # base_sensor_rot_offset = [1, 0.3, 0.9, 0]

    # global_gain = [20]
    # gain = [700, 900, 908]
    # per_channel_gain = [500, 500, 500]
    # bias = [0.1, 0.1, 0.1]
    # noise = [2, 2, 2]
    # sensor_offset = [-4, 8, 8]
    # sensor_rot_offset = [1, 0, 0.6, 0]  # x, y, z, w
    # ring_offset = [8, 9, 9]
    # ring_rot_offset = [1, 0.3, 0.7, 0.2]  # x, y, z, w
    # crosstalk = [3, 2, 4]
    # base_sensor_offset = [0.2, 0.4, 0.5]
    # base_sensor_rot_offset = [1, 0.3, 0.9, 0]

    # 将需要优化的参数打包
    ####################################
    # pos = np.array([5.0, 6.0, 7.0])
    # rot = np.array([1.0, 0.0, 0.0, 0.0])
    # x0 = encode_x(global_gain, gain, per_channel_gain, bias, noise, sensor_offset, sensor_rot_offset,
    #               ring_offset, ring_rot_offset, crosstalk, base_sensor_offset, base_sensor_rot_offset)
    # sensorsresult = compute_sensor_newly(pos, rot, x0)
    # print(sensorsresult)
    #####################################
    # if SHOW_FIGURE:
    #     plt.figure()
    #     # plt.plot(mag_data,c="r")
    #     ax1 = plt.axes(projection='3d')
    #     # ax1.scatter3D(mag_data[:,0], mag_data[:,1], mag_data[:,2], c='red')
    #     # ax1.scatter3D(mag_data[:,3], mag_data[:,4], mag_data[:,5], c='blue')
    #     # ax1.scatter3D(mag_data[:,6], mag_data[:,7], mag_data[:,8], c='yellow')
    #     ax1.scatter3D(sensorsresult[:, 0], sensorsresult[:, 1], sensorsresult[:, 2], c='yellow')
    #     # plt.plot(pos_train)
    #     plt.title("TRAIN")
    #     plt.show()


    # main()
