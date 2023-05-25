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

# 计算误差值
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


# 将多余的点剔除出去
def filter_bad_data(data, use_abs=False):
    print(data.shape)
    if use_abs:
        transform = lambda x: np.abs(x)
    else:
        transform = lambda x: x

    filtered = scipy.signal.savgol_filter(transform(data), 15, 2, axis=0)
    error = np.sqrt(np.sum((transform(data)-filtered)**2, axis=1))

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
    # 此处切换为欧拉角
    sensor_rot_offset = data[i:i + 3]  # x, y, z,
    i += 3
    ring_offset = data[i:i + 3]
    i += 3
    ring_rot_offset = data[i:i + 3]
    i += 3
    crosstalk = data[i:i + 3]
    i += 3
    base_sensor_offset = data[i:i + 3]
    i += 3
    base_sensor_rot_offset = data[i:i + 3]
    i += 3
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

# 设定学习参数的范围
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


#用于剔除无效的数据
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

# 加载我们用机械臂拿到的数据
def load_resampled_data_new(path):
    all_data = np.loadtxt(path)
    mag_data = all_data[:, 6:9]
    # mag_data = mag_data[:, [0, 2, 1]]#可能不需要调整
    pos = all_data[:, 0:3]
    rot = all_data[:, 3:6]
    return mag_data, pos, rot


# 磁偶极子根据空间位置的预期读书
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
    kxy,kxz,kyz =0.,0.,0.

    # 计算场强值（改成旋转矩阵形式）
    ring_q =rot
    #旋转矩阵运算和apply运算的结果需要转置
    sensor = base_sensor_offset + sensor_offset

    # 改成旋转矩阵形式
    # sensor_rot = sensor_rot_offset * base_sensor_rot_offset
    sensor_rot_offset = R.from_euler('xyz', sensor_rot_offset, degrees=True).as_matrix()
    base_sensor_rot_offset = R.from_euler('xyz', base_sensor_rot_offset, degrees=True).as_matrix()
    sensor_rot = np.matmul(sensor_rot_offset,base_sensor_rot_offset)

    # 改成欧拉角和旋转矩阵
    # ring_qs = R.from_quat(ring_q)
    ring_qs = R.from_euler('xyz', ring_q, degrees=True).as_matrix()
    # ring_pos_adj = pos + ring_qs.apply(ring_offset, inverse=True)
    ring_pos_adj = pos + np.matmul(np.linalg.inv(ring_qs) ,ring_offset)
    # ring_rot_adj = ring_qs * ring_rot_offset
    ring_rot_offset = R.from_euler('xyz', ring_rot_offset, degrees=True).as_matrix()
    ring_rot_adj = np.matmul(ring_qs,ring_rot_offset)

    # sensor_ring = (ring_qs * ring_rot_offset).apply(sensor - ring_pos_adj)
    vetornew = (sensor - ring_pos_adj).reshape(-1,3,1)
    sensor_ring = np.matmul(ring_rot_adj,vetornew).reshape(-1,3)
    # field = ring_rot_adj.apply(dipole_model_newly(sensor_ring), inverse=True)
    dipolevalue = dipole_model_newly(sensor_ring).reshape(-1,3,1)
    field = np.matmul(np.linalg.inv(ring_rot_adj) ,dipolevalue)
    # field = sensor_rot.apply(field)
    field = np.matmul(sensor_rot,field).reshape(-1,3)

    # 计算根据公式计算理论的预测读数值
    field_adj = field * global_gain / per_channel_gain
    # 此处的公式和Aruraring的差异较大一个用的field一个用的field_adj,同时引入了更多的协防差
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
    # 对应的是公式
    sensors[sensors < 0] = 0

    return sensors


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


def main():
    # 加载实际的Sensor的读数以及对应的Pose和Rotation数据
    dataPath = "/home/gaofei/Aura/Aruadata/reciver_to_sender_6dofwithmag.txt"
    # needchange = True
    needchange = False
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
        calibrated_new = compute_sensor_newly(pos, rot, x)*0.1
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

    # 将所有的四元数变为欧拉角
    sensor_rot_offset =[120,60,20]
    ring_offset = [8,9,9]
    ring_rot_offset =[42,45,90]
    crosstalk = [3,2,4]
    base_sensor_offset = [0.2,0.4,0.5]
    base_sensor_rot_offset = [20,39,28]

    # 将需要优化的参数打包
    x0 = encode_x(global_gain,gain,per_channel_gain,bias,noise,sensor_offset,sensor_rot_offset,
                  ring_offset,ring_rot_offset,crosstalk,base_sensor_offset,base_sensor_rot_offset)
    optionsmap = {"maxiter": 10,'disp': True}
    print("Running minimizer...")
    # x_opt = scipy.optimize.minimize(costBiRing, x0=x0, bounds=get_bounds())

    # x_opt = scipy.optimize.minimize(costBiRing, x0=x0,options=optionsmap,bounds=get_bounds_new())
    x_opt = scipy.optimize.minimize(costBiRing, x0=x0,options=optionsmap)
    # print("final result is :",x_opt.x)

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
    # calibrated_new = compute_sensor_newly(pos, rot, x0)

    # caculate_init = compute_sensor_newly(pos, rot, x0)
    #
    # print(caculate_init)

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    # generating  random dataset
    # z = np.random.randint(80, size=(55))
    # x = np.random.randint(60, size=(55))
    # y = np.random.randint(64, size=(55))
    # Creating figures for the plot
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating a plot using the random datasets
    np.savetxt("/home/gaofei/BiRing/caculatedresult/caculated.txt",calibrated_new)
    calibratex =calibrated_new[:,0]*10
    calibratey =calibrated_new[:,1]*10
    calibratez =calibrated_new[:,2]*10
    ax.scatter3D(calibratex, calibratey, calibratez, color="red")
    plt.title("3D caculated plot")
    # display the  plot
    # plt.show()
    oldx = mag_data[:, 0]
    oldy = mag_data[:, 1]
    oldz = mag_data[:, 2]
    ax.scatter3D(oldx, oldy, oldz, color="blue")
    # display the  plot
    plt.show()

    # print("calibrated :",calibrated_new)
    # print("real mag :",mag_data)



#1.改动替换四元数运算为矩阵运算即可

if __name__ == "__main__":
    main()
    # import matplotlib.pyplot as plt
    # from mpl_toolkits import mplot3d
    #
    # # generating  random dataset
    # # z = np.random.randint(80, size=(55))
    # # x = np.random.randint(60, size=(55))
    # # y = np.random.randint(64, size=(55))
    # # Creating figures for the plot
    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection="3d")
    # # Creating a plot using the random datasets
    # calibrated_new = np.loadtxt("/home/gaofei/BiRing/caculatedresult/caculated.txt")
    # calibratex = calibrated_new[:, 0]
    # calibratey = calibrated_new[:, 1]
    # calibratez = calibrated_new[:, 2]
    # ax.scatter3D(calibratex, calibratey, calibratez, color="red")
    # # oldx = mag_data[:, 0]
    # # oldy = mag_data[:, 1]
    # # oldz = mag_data[:, 2]
    # # ax.scatter3D(oldx, oldy, oldz, color="blue")
    # plt.title("3D scatter plot")
    # # display the  plot
    # plt.show()