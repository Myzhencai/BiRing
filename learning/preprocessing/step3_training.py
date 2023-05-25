import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
import seaborn as sns
from sklearn.neural_network import MLPRegressor

from mag_utils import DeviceCalibration, compute_sensor, get_field, plot_correlation, get_sensor_pos
from utils import load_resampled_data, save_mode, save_pos_data, save_predictions
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)

TRIAL = "par5_1"
# TRIAL_SAVE = "1245"
TRIAL = ["par1_2", "par2_2", "par5_2", "par4_1", "par6_2", "par3_2", "par7_1", "par8_2"]
TRIAL = ["exp10"]
TRIAL_CALIBRATE = 'exp11_'
TEST_FRAC = 0.1
NDOF = 3


CUTOFF = 500
state = DeviceCalibration.from_trial(TRIAL_CALIBRATE)
# USE_BIAS = True
USE_BIAS = False
SHOW_FIGURE = True


def split_data(data):
    split_point = round(len(data) * (1 - TEST_FRAC))
    data_train = data[0:split_point, :]
    data_test = data[split_point:, :]
    return data_train, data_test


def train_model(x_train, y_train, x_test, y_test):
    # regr = MLPRegressor(hidden_layer_sizes=(20,60,80,160,120,80,60,30,20), random_state=1, max_iter=1400000000,activation="relu",solver='adam', verbose=False).fit(x_train, y_train)  #20
    regr = MLPRegressor(hidden_layer_sizes=(20,60,80,160,120,80,60,30,20), max_iter=140,activation="relu",solver='adam').fit(x_train, y_train)  #20
    # regr = MLPRegressor(hidden_layer_sizes=(80,164,68,30),activation="relu",solver='adam',learning_rate_init=0.001,power_t=0.5,alpha=0.0001,
    #                     random_state=1, max_iter=140, verbose=False).fit(x_train, y_train)  #20
    y_train_predict = regr.predict(x_train)
    y_test_predict = regr.predict(x_test)
    error_train = y_train_predict - y_train
    error_test = y_test_predict - y_test

    mean_error_train = np.mean(np.sqrt(np.sum(np.square(error_train), axis=1)))
    mean_error_test = np.mean(np.sqrt(np.sum(np.square(error_test), axis=1)))
    # mean_error_train = np.sum(np.sqrt(np.sum(np.square(error_train), axis=1)))
    # mean_error_test = np.sum(np.sqrt(np.sum(np.square(error_test), axis=1)))
    print(mean_error_train)
    print(mean_error_test)

    if SHOW_FIGURE:
        plt.figure("trained")
        plt.title("result")
        plt.plot(y_train_predict,c="r")
        plt.plot(y_train)

        plt.figure("tested")
        plt.title("testresult")
        plt.plot(y_test_predict,c ="r")
        plt.plot(y_test)
        plt.show()
    return regr, y_train_predict, y_test_predict


def main():
    mag_data_all = []
    pos_all = []
    rot_all = []
    mean_error_all = []

    for trial_test in TRIAL:
        trial_train = TRIAL.copy()
        # trial_train = TRIAL
        # trial_train.remove(trial_test)
        print("Training on: ", trial_train)
        print("Testing on: ", trial_test)

        for user in trial_train:
            mag_data, pos, rot = load_resampled_data(user)
            # mag_data = mag_data[:-CUTOFF]
            # pos = pos[:-CUTOFF]
            # rot = rot[:-CUTOFF]
            mag_data = mag_data
            pos = pos
            rot = rot
            try:
                mag_data_all = np.append(mag_data_all, mag_data, axis=0)
                pos_all = np.append(pos_all, pos, axis=0)
                rot_all = np.append(rot_all, rot, axis=0)
            except:
                mag_data_all = mag_data
                pos_all = pos
                rot_all = rot
            # sensor_pos = get_sensor_pos(pos, rot, state.coil1)
            # sensor_pos_train, sensor_pos_test = split_data(sensor_pos)

        pos_train, pos_test = split_data(pos_all)
        rot_train, rot_test = split_data(rot_all)
        mag_train, mag_test = split_data(mag_data_all)

        sensor_train = compute_sensor(get_field(pos_train, rot_train, state.coil1), state.coil1)
        # 将此处直接换成元书的magdata
        sensor_test = compute_sensor(get_field(pos_test, rot_test, state.coil1), state.coil1)
        #替换成原始的mag数据
        if SHOW_FIGURE:
            plt.figure()
            plt.plot(mag_train)
            plt.figure()
            plt.plot(sensor_train)
            plot_correlation(sensor_train, mag_train)

        model_mag2sens, sensors_pred_train, sensors_pred_test = train_model(mag_train, sensor_train, mag_test, sensor_test)

        if NDOF == 2:
            pos_train = pos_train[:, :-1]
            pos_test = pos_test[:, :-1]

        model_sens2pos, pos_pred_train, pos_pred_test = train_model(sensor_train, pos_train, sensor_test, pos_test)

        pos_pred_train = model_sens2pos.predict(sensors_pred_train)
        pos_pred_test = model_sens2pos.predict(sensors_pred_test)

        error_train = pos_pred_train - pos_train
        error_test = pos_pred_test - pos_test

        mean_error_train = np.mean(np.sqrt(np.sum(np.square(error_train), axis=1)))
        mean_error_test = np.mean(np.sqrt(np.sum(np.square(error_test), axis=1)))
        print("Mean train error (mm): ", mean_error_train)
        print("Mean test error (mm): ", mean_error_test)

        if SHOW_FIGURE:
            plt.figure()
            plt.plot(pos_pred_train)
            plt.plot(pos_train)
            plt.title("TRAIN")

            plt.figure()
            plt.plot(pos_pred_test)
            plt.plot(pos_test)
            plt.title("TEST")
            plt.show()

        save_mode(model_mag2sens, "mag2sensor", TRIAL, TRIAL_CALIBRATE, NDOF)
        save_mode(model_sens2pos, "sensor2pos", TRIAL, TRIAL_CALIBRATE, NDOF)

        mag_data, pos, rot = load_resampled_data(trial_test)
        mag_data = mag_data[:-CUTOFF]
        pos = pos[:-CUTOFF]
        rot = rot[:-CUTOFF]
        sensor_predict = model_mag2sens.predict(mag_data)
        pos_predict = model_sens2pos.predict(sensor_predict)
        if NDOF == 2:
            pos = pos[:, :-1]
        if USE_BIAS:
            pos_initial = np.mean(pos[:50], axis=0)
            pos_predict_initial = np.mean(pos_predict[:50], axis=0)
            bias = pos_initial - pos_predict_initial
            pos_predict += bias

        error = pos_predict - pos
        dummy_pos = np.mean(pos, axis=0)
        dummy_pos = dummy_pos.reshape(1, np.size(dummy_pos, 0))
        dummy_pos = np.repeat(dummy_pos, np.size(pos, 0), axis=0)
        dummy_pos_error = dummy_pos - pos

        mean_error = np.mean(np.abs(error), axis=0)
        print("Mean error (mm): ", mean_error)
        print("Mean absolute error (mm): ", np.linalg.norm(mean_error))
        mean_sqaure_error = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
        print("Mean square absolute error (mm): ", mean_sqaure_error)
        dummy_mean_error = np.mean(np.abs(dummy_pos_error), axis=0)
        print("Baseline test error: ", dummy_mean_error)
        print("Mean absolute error (mm): ", np.linalg.norm(dummy_mean_error))
        mean_error_all.append(mean_error)

        if SHOW_FIGURE:
            plt.figure()
            plt.plot(pos)
            plt.plot(pos_predict)
            plt.title(trial_test)
            plt.show()

        save_pos_data(trial_train, trial_test, pos, str(NDOF))
        save_predictions(trial_train, trial_test, pos_predict, str(NDOF))
    # pickle.dump(model_mag2sens, open("mag2sensor_"+TRIAL+".sav", 'wb'))
    # pickle.dump(model_sens2pos, open("sensor2pos_"+TRIAL+".sav", 'wb'))

    print("Cross User Mean Absolute Error (mm):", np.mean(mean_error_all, axis=0))

def load_real_magdata(path):
    file = path
    all_data = np.loadtxt(file)
    mag_data = all_data[:, 6:15]
    # mag_data = all_data[:, 6:12]
    # mag_data = mag_data[:, [0, 2, 1]]
    pos = all_data[:, 0:3]
    rot = all_data[:, 3:6]
    return mag_data, pos, rot


def save_pos_data_txt(path,data):
    np.savetxt(path,data)


def voltega2pos():
    mag_data_all = []
    pos_all = []
    rot_all = []
    mean_error_all = []

    # real_magdata_path = "/home/gaofei/BiRing/Data/processed__t1.txt"
    real_magdata_path = "/home/gaofei/Aura/Aruadata/reciver_to_sender_6dofwithmag.txt.txt"
    mag_data, pos, rot = load_real_magdata(real_magdata_path)
    # CUTOFF =1000
    # mag_data = mag_data[:-CUTOFF]
    # pos = pos[:-CUTOFF]
    # rot = rot[:-CUTOFF]
    try:
        mag_data_all = np.append(mag_data_all, mag_data, axis=0)
        pos_all = np.append(pos_all, pos, axis=0)
        rot_all = np.append(rot_all, rot, axis=0)
    except:
        mag_data_all = mag_data
        pos_all = pos
        rot_all = rot
            # sensor_pos = get_sensor_pos(pos, rot, state.coil1)
            # sensor_pos_train, sensor_pos_test = split_data(sensor_pos)

    pos_train, pos_test = split_data(pos_all)
    # rot_train, rot_test = split_data(rot_all)
    mag_train, mag_test = split_data(mag_data_all)

    if SHOW_FIGURE:
        plt.figure()
        plt.title("mag_train")
        plt.plot(mag_train)

    if NDOF == 2:
        pos_train = pos_train[:, :-1]
        pos_test = pos_test[:, :-1]


    # net = Net(n_features=9, n_hidden=80, n_output=3)
    #
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    # loss_func = torch.nn.MSELoss()
    #
    # plt.ion()
    #
    # for t in range(100):
    #     mag_train = torch.tensor(mag_train)
    #     pos_train = torch.tensor(pos_train)
    #     predict = net(mag_train)
    #
    #     loss = loss_func(predict, pos_train)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

        # if t % 10 == 0:
        #     plt.cla()
        #     plt.scatter(x.data.numpy(), y.data.numpy())
        #     plt.plot(x.data.numpy(), y.data.numpy(), 'r-', lw=5)
        #     plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        #     plt.show()
        #     plt.pause(0.1)
        #
        # plt.ioff()

    model_realmag2pos, pos_pred_train, pos_pred_test = train_model(mag_train, pos_train, mag_test, pos_test)

    pos_pred_train = model_realmag2pos.predict(mag_train)
    pos_pred_test = model_realmag2pos.predict(mag_test)

    error_train = pos_pred_train - pos_train
    error_test = pos_pred_test - pos_test

    mean_error_train = np.mean(np.sqrt(np.sum(np.square(error_train), axis=1)))
    mean_error_test = np.mean(np.sqrt(np.sum(np.square(error_test), axis=1)))
    print("Mean train error (mm): ", mean_error_train)
    print("Mean test error (mm): ", mean_error_test)

    if SHOW_FIGURE:
        plt.figure()
        plt.plot(pos_pred_train,c="r")
        plt.plot(pos_train)
        plt.title("TRAIN")

        plt.figure()
        plt.plot(pos_pred_test,c="r")
        plt.plot(pos_test)
        plt.title("TEST")
        plt.show()
    save_mode(model_realmag2pos, "model_realmag2pos", TRIAL, TRIAL_CALIBRATE, NDOF)

    # 检测新数据
    real_magdata_path_new = "/home/gaofei/Aura/Aruadata/test_4traindata.txt"
    mag_data, pos, rot = load_real_magdata(real_magdata_path_new)
    mag_data = mag_data
    pos = pos
    rot = rot
    pos_predict = model_realmag2pos.predict(mag_data)
    if NDOF == 2:
        pos = pos[:, :-1]
    if USE_BIAS:
        pos_initial = np.mean(pos[:50], axis=0)
        pos_predict_initial = np.mean(pos_predict[:50], axis=0)
        bias = pos_initial - pos_predict_initial
        pos_predict += bias

    error = pos_predict - pos
    dummy_pos = np.mean(pos, axis=0)
    dummy_pos = dummy_pos.reshape(1, np.size(dummy_pos, 0))
    dummy_pos = np.repeat(dummy_pos, np.size(pos, 0), axis=0)
    dummy_pos_error = dummy_pos - pos

    mean_error = np.mean(np.abs(error), axis=0)
    print("Mean error (mm): ", mean_error)
    print("Mean absolute error (mm): ", np.linalg.norm(mean_error))
    mean_sqaure_error = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
    print("Mean square absolute error (mm): ", mean_sqaure_error)
    dummy_mean_error = np.mean(np.abs(dummy_pos_error), axis=0)
    print("Baseline test error: ", dummy_mean_error)
    print("Mean absolute error (mm): ", np.linalg.norm(dummy_mean_error))
    mean_error_all.append(mean_error)

    if SHOW_FIGURE:
        plt.figure()
        plt.plot(pos)
        plt.plot(pos_predict,c="r")
        plt.title("newly test ")
        plt.show()

    save_pos_data_txt("/home/gaofei/BiRing/Data/posedata/posedatareal.txt", pos)
    save_pos_data_txt("/home/gaofei/BiRing/Data/posedata/posedatapredicet.txt", pos_predict)
    # pickle.dump(model_mag2sens, open("mag2sensor_"+TRIAL+".sav", 'wb'))
    # pickle.dump(model_sens2pos, open("sensor2pos_"+TRIAL+".sav", 'wb'))

    print("Cross User Mean Absolute Error (mm):", np.mean(mean_error_all, axis=0))

import matplotlib.pyplot as plt
import numpy as np
import numpy.random

def creat3d_vec_xyz(start_points, end_points, colors=None, view_angle=20, azimuth=30):
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
        colors = numpy.random.randint(0, 255, size=start_points.shape, dtype=np.uint8)
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

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    ax.view_init(view_angle, azimuth)
    ax.set_title('coordinates-xyz')
    plt.show()


def creat3d_vec_xyz(start_points, end_points, end_points1, end_points2, colors=None, view_angle=20, azimuth=30):
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
        colors = numpy.random.randint(0, 255, size=start_points.shape, dtype=np.uint8)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = fig.gca(projection='3d')
    num_vec = start_points.shape[0]

    # q = ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2], end_points[:, 0], end_points[:, 1],
    #               end_points[:, 2], color="#666666", arrow_length_ratio=0.1)
    for i in range(num_vec):
        color = '#'
        for j in range(3):
            color += str(hex(colors[i, j]+5))[-2:].replace('x', '0').upper()
        q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points[i, 0], end_points[i, 1],
                      end_points[i, 2], color=color, arrow_length_ratio=0.1)

    for i in range(num_vec):
        color = '#'
        for j in range(3):
            color += str(hex(colors[i, j]))[-2:].replace('x', '0').upper()
        q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points1[i, 0], end_points1[i, 1],
                      end_points1[i, 2], color=color, arrow_length_ratio=0.1)

    for i in range(num_vec):
        color = '#'
        for j in range(3):
            color += str(hex(colors[i, j]-5))[-2:].replace('x', '0').upper()
        q = ax.quiver(start_points[i, 0], start_points[i, 1], start_points[i, 2], end_points2[i, 0], end_points2[i, 1],
                      end_points2[i, 2], color=color, arrow_length_ratio=0.1)

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    ax.view_init(view_angle, azimuth)
    ax.set_title('coordinates-xyz')
    plt.show()


def voltega2pos():
    mag_data_all = []
    pos_all = []
    rot_all = []
    mean_error_all = []

    # real_magdata_path = "/home/gaofei/BiRing/Data/processed__t1.txt"
    real_magdata_path = "/home/gaofei/Aura/Aruadata/test_6traindata.txt"
    mag_data, pos, rot = load_real_magdata(real_magdata_path)

    if SHOW_FIGURE:
        plt.figure()
        # plt.plot(mag_data,c="r")
        ax1 = plt.axes(projection='3d')
        # ax1.scatter3D(mag_data[:,0], mag_data[:,1], mag_data[:,2], c='red')
        # ax1.scatter3D(mag_data[:,3], mag_data[:,4], mag_data[:,5], c='blue')
        # ax1.scatter3D(mag_data[:,6], mag_data[:,7], mag_data[:,8], c='yellow')
        ax1.scatter3D(rot[:,0], rot[:,1], rot[:,2], c='yellow')
        # plt.plot(pos_train)
        plt.title("TRAIN")
        plt.show()


        # plt.figure()
        # plt.plot(pos_pred_test,c="r")
        # plt.plot(pos_test)
        # plt.title("TEST")
        # plt.show()
    # CUTOFF =1000
    # mag_data = mag_data[:-CUTOFF]
    # pos = pos[:-CUTOFF]
    # rot = rot[:-CUTOFF]
    try:
        mag_data_all = np.append(mag_data_all, mag_data, axis=0)
        pos_all = np.append(pos_all, pos, axis=0)
        rot_all = np.append(rot_all, rot, axis=0)
    except:
        mag_data_all = mag_data
        pos_all = pos
        rot_all = rot
            # sensor_pos = get_sensor_pos(pos, rot, state.coil1)
            # sensor_pos_train, sensor_pos_test = split_data(sensor_pos)

    pos_train, pos_test = split_data(pos_all)
    # rot_train, rot_test = split_data(rot_all)
    mag_train, mag_test = split_data(mag_data_all)

    if SHOW_FIGURE:
        plt.figure()
        plt.title("mag_train")
        plt.plot(mag_train)

    if NDOF == 2:
        pos_train = pos_train[:, :-1]
        pos_test = pos_test[:, :-1]


    # net = Net(n_features=9, n_hidden=80, n_output=3)
    #
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    # loss_func = torch.nn.MSELoss()
    #
    # plt.ion()
    #
    # for t in range(100):
    #     mag_train = torch.tensor(mag_train)
    #     pos_train = torch.tensor(pos_train)
    #     predict = net(mag_train)
    #
    #     loss = loss_func(predict, pos_train)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

        # if t % 10 == 0:
        #     plt.cla()
        #     plt.scatter(x.data.numpy(), y.data.numpy())
        #     plt.plot(x.data.numpy(), y.data.numpy(), 'r-', lw=5)
        #     plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        #     plt.show()
        #     plt.pause(0.1)
        #
        # plt.ioff()

    model_realmag2pos, pos_pred_train, pos_pred_test = train_model(mag_train, pos_train, mag_test, pos_test)

    pos_pred_train = model_realmag2pos.predict(mag_train)
    pos_pred_test = model_realmag2pos.predict(mag_test)

    error_train = pos_pred_train - pos_train
    error_test = pos_pred_test - pos_test

    mean_error_train = np.mean(np.sqrt(np.sum(np.square(error_train), axis=1)))
    mean_error_test = np.mean(np.sqrt(np.sum(np.square(error_test), axis=1)))
    print("Mean train error (mm): ", mean_error_train)
    print("Mean test error (mm): ", mean_error_test)

    if SHOW_FIGURE:
        plt.figure()
        plt.plot(pos_pred_train,c="r")
        plt.plot(pos_train)
        plt.title("TRAIN")

        plt.figure()
        plt.plot(pos_pred_test,c="r")
        plt.plot(pos_test)
        plt.title("TEST")
        plt.show()
    save_mode(model_realmag2pos, "model_realmag2pos", TRIAL, TRIAL_CALIBRATE, NDOF)

    # 检测新数据
    real_magdata_path_new = "/home/gaofei/Aura/Aruadata/test_4traindata.txt"
    mag_data, pos, rot = load_real_magdata(real_magdata_path_new)
    mag_data = mag_data
    pos = pos
    rot = rot
    pos_predict = model_realmag2pos.predict(mag_data)
    if NDOF == 2:
        pos = pos[:, :-1]
    if USE_BIAS:
        pos_initial = np.mean(pos[:50], axis=0)
        pos_predict_initial = np.mean(pos_predict[:50], axis=0)
        bias = pos_initial - pos_predict_initial
        pos_predict += bias

    error = pos_predict - pos
    dummy_pos = np.mean(pos, axis=0)
    dummy_pos = dummy_pos.reshape(1, np.size(dummy_pos, 0))
    dummy_pos = np.repeat(dummy_pos, np.size(pos, 0), axis=0)
    dummy_pos_error = dummy_pos - pos

    mean_error = np.mean(np.abs(error), axis=0)
    print("Mean error (mm): ", mean_error)
    print("Mean absolute error (mm): ", np.linalg.norm(mean_error))
    mean_sqaure_error = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
    print("Mean square absolute error (mm): ", mean_sqaure_error)
    dummy_mean_error = np.mean(np.abs(dummy_pos_error), axis=0)
    print("Baseline test error: ", dummy_mean_error)
    print("Mean absolute error (mm): ", np.linalg.norm(dummy_mean_error))
    mean_error_all.append(mean_error)

    if SHOW_FIGURE:
        plt.figure()
        plt.plot(pos)
        plt.plot(pos_predict,c="r")
        plt.title("newly test ")
        plt.show()

    save_pos_data_txt("/home/gaofei/BiRing/Data/posedata/posedatareal.txt", pos)
    save_pos_data_txt("/home/gaofei/BiRing/Data/posedata/posedatapredicet.txt", pos_predict)
    # pickle.dump(model_mag2sens, open("mag2sensor_"+TRIAL+".sav", 'wb'))
    # pickle.dump(model_sens2pos, open("sensor2pos_"+TRIAL+".sav", 'wb'))

    print("Cross User Mean Absolute Error (mm):", np.mean(mean_error_all, axis=0))
#
# class Net(torch.nn.Module):
#     def __init__(self, n_features, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_features, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x




if __name__ == "__main__":
    # main()
    real_magdata_path_new = "/home/gaofei/Aura/Aruadata/test_6traindata.txt"
    mag_data, pos, rot = load_real_magdata(real_magdata_path_new)


    num = 50
    end_points = mag_data[:,:3][:num]*0.001
    end_points1 = mag_data[:,3:6][:num]*0.001
    end_points2 = mag_data[:,6:9][:num]*0.001
    # end_points = mag_data[:,:3]*0.001
    # end_points = np.array(end_points)
    # end_points = end_points / np.sqrt(np.sum(end_points ** 2, axis=1)).reshape(-1, 1)
    # start_points = pos[:1000]
    start_points = np.zeros((num,3))
    creat3d_vec_xyz(start_points, end_points,end_points1,end_points2)
    # creat3d_vec_xyz(start_points, end_points1)
    # creat3d_vec_xyz(start_points, end_points2)
    # voltega2pos()