"""
@author: LXA
 Data: 2020 年 5 月 31 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import DNN_data
import time
import CPDNN_base
import DNN_base
import DNN_tools
import pLaplace_eqs1d
import laplace_eqs1d
import plotData
import saveData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout, actName=None):
    DNN_tools.log_string('Laplace type for problem: %s\n' % (R_dic['laplace_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)

    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(actName), log_fileout)

    if R['laplace_type'] == 'p_laplace':
        DNN_tools.log_string('order of p-Laplacian: %s\n' % (R_dic['order2laplace']), log_fileout)
        DNN_tools.log_string('epsilon: %f\n' % (R_dic['epsilon']), log_fileout)

    if R_dic['variational_loss'] == 1:
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: original function loss\n', log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


def solve_laplace(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name1 = '%s_%s.txt' % ('log2train', R['act_name2NN1'])
    log_fileout_NN1 = open(os.path.join(log_out_path, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout_NN1, actName=R['act_name2NN1'])

    outfile_name = '%s_%s.txt' % ('log2train', R['act_name2NN2'])
    log_fileout_NN2 = open(os.path.join(log_out_path, outfile_name), 'w')
    dictionary_out2file(R, log_fileout_NN2, actName=R['act_name2NN2'])

    # laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    bd_penalty_init = R['boundary_penalty']         # Regularization parameter for boundary conditions
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']
    wb_regular = R['regular_weight_biases']         # Regularization parameter for weights and biases

    # ------- set the problem ---------
    input_dim = R['input_dim']
    out_dim = R['output_dim']

    region_l = 0.0
    region_r = 1.0
    if R['laplace_type'] == 'general_laplace':
        # -laplace u = f
        region_l = 0.0
        region_r = 1.0
        f, u_true, u_left, u_right = laplace_eqs1d.get_laplace_infos(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_l, right_top=region_r, laplace_name=R['equa_name'])
    elif R['laplace_type'] == 'p_laplace':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
        p_index = R['order2laplace']
        epsilon = R['epsilon']
        # index2freq = R['flag2freq'] #一直报错，为什么？
        index2freq = [1, 1.0 / epsilon]
        if 2 == p_index:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_2laplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon)
        elif 3 == p_index:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_3laplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon)
        elif 5 == p_index:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_5laplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon)
        elif 8 == p_index:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_8laplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon)
        else:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_pLaplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon,
                eqs_name=R['eqs_name'])

    # 初始化权重和和偏置的模式
    if R['weight_biases_model'] == 'general_model':
        flag1 = 'WB_NN1'
        flag2 = 'WB_NN2'
        # Weights, Biases = PDE_DNN_base.Initial_DNN2different_hidden(input_dim, out_dim, hidden_layers, flag)
        # Weights, Biases = laplace_DNN1d_base.initialize_NN_xavier(input_dim, out_dim, hidden_layers, flag1)
        # Weights, Biases = laplace_DNN1d_base.initialize_NN_random_normal(input_dim, out_dim, hidden_layers, flag1)
        Weights_NN1, Biases_NN1 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag1)
        Weights_NN2, Biases_NN2 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2)
    elif R['weight_biases_model'] == 'phase_shift_model':  # phase_shift 这个还需要研究，先放在这里
        flag0X = 'X_WB0'
        Weights0_X, Biases0_X = CPDNN_base.Initial_DNN(input_dim, out_dim, hidden_layers, flag0X)

        flag0Y = 'Y_WB0'
        Weights0_Y, Biases0_Y = CPDNN_base.Initial_DNN(input_dim, out_dim, hidden_layers, flag0Y)

        # Weights_COS， Weights_SIN， Biases_COS 和 Biases_SIN 对应m=1,2.3.....的项
        Weights_COS_X = []
        Weights_SIN_X = []
        Biases_COS_X = []
        Biases_SIN_X = []

        Weights_COS_Y = []
        Weights_SIN_Y = []
        Biases_COS_Y = []
        Biases_SIN_Y = []
        # 一个频率对应一组 Weights 和 Biases(分 Cos 和 Sin)
        flag1X = ''
        flag2X = ''
        for k in range(len(index2freq)):
            flag1X = 'X_WB' + '2Cos-' + str(index2freq[k])
            Weights_X, Biases_X = CPDNN_base.Initial_DNN(input_dim, out_dim, hidden_layers, flag1X)
            Weights_COS_X.append(Weights_X)
            Biases_COS_X.append(Biases_X)

            flag2X = 'X_WB' + '2Sin-' + str(index2freq[k])
            Weights_X, Biases_X = CPDNN_base.Initial_DNN(input_dim, out_dim, hidden_layers, flag2X)
            Weights_SIN_X.append(Weights_X)
            Biases_SIN_X.append(Biases_X)

        for k in range(len(index2freq)):
            flag1Y = 'Y_WB' + '2Cos-' + str(index2freq[k])
            Weights_Y, Biases_Y = CPDNN_base.Initial_DNN(input_dim, out_dim, hidden_layers, flag1Y)
            Weights_COS_Y.append(Weights_Y)
            Biases_COS_Y.append(Biases_Y)

            flag2Y = 'Y_WB' + '2Sin-' + str(index2freq[k])
            Weights_Y, Biases_Y = CPDNN_base.Initial_DNN(input_dim, out_dim, hidden_layers, flag2Y)
            Weights_SIN_Y.append(Weights_Y)
            Biases_SIN_Y.append(Biases_Y)

    act_func1 = R['act_name2NN1']
    act_func2 = R['act_name2NN2']
    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X_it = tf.placeholder(tf.float32, name='X_it', shape=[None, input_dim])                # * 行 1 列
            X_left_bd = tf.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])      # * 行 1 列
            X_right_bd = tf.placeholder(tf.float32, name='X_right_bd', shape=[None, input_dim])    # * 行 1 列
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            # 供选择的网络模式
            if R['model'] == 'PDE_DNN':
                U_NN1 = DNN_base.PDE_DNN(X_it, Weights_NN1, Biases_NN1, hidden_layers, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN(X_left_bd, Weights_NN1, Biases_NN1, hidden_layers, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN(X_right_bd, Weights_NN1, Biases_NN1, hidden_layers, activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN(X_it, Weights_NN2, Biases_NN2, hidden_layers, activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN(X_left_bd, Weights_NN2, Biases_NN2, hidden_layers, activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN(X_right_bd, Weights_NN2, Biases_NN2, hidden_layers, activate_name=act_func2)
            elif R['model'] == 'PDE_DNN_BN':
                U_NN1 = DNN_base.PDE_DNN_BN(X_it, Weights_NN1, Biases_NN1, hidden_layers, activate_name=act_func1,
                                            is_training=train_opt)
                ULeft_NN1 = DNN_base.PDE_DNN_BN(X_left_bd, Weights_NN1, Biases_NN1, hidden_layers, activate_name=act_func1,
                                                is_training=train_opt)
                URight_NN1 = DNN_base.PDE_DNN_BN(X_right_bd, Weights_NN1, Biases_NN1, hidden_layers, activate_name=act_func1,
                                                 is_training=train_opt)

                U_NN2 = DNN_base.PDE_DNN_BN(X_it, Weights_NN2, Biases_NN2, hidden_layers, activate_name=act_func2,
                                            is_training=train_opt)
                ULeft_NN2 = DNN_base.PDE_DNN_BN(X_left_bd, Weights_NN2, Biases_NN2, hidden_layers, activate_name=act_func2,
                                                is_training=train_opt)
                URight_NN2 = DNN_base.PDE_DNN_BN(X_right_bd, Weights_NN2, Biases_NN2, hidden_layers, activate_name=act_func2,
                                                is_training=train_opt)
            elif R['model'] == 'PDE_DNN_scale':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN1 = DNN_base.PDE_DNN_scale(X_it, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN_scale(X_left_bd, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN_scale(X_right_bd, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN_scale(X_it, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN_scale(X_left_bd, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN_scale(X_right_bd, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
            elif R['model'] == 'PDE_DNN_adapt_scale':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN1 = DNN_base.PDE_DNN_adapt_scale(X_it, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN_adapt_scale(X_left_bd, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN_adapt_scale(X_right_bd, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN_adapt_scale(X_it, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN_adapt_scale(X_left_bd, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN_adapt_scale(X_right_bd, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
            elif R['model'] == 'PDE_DNN_FourierBase':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN1 = DNN_base.PDE_DNN_FourierBase(X_it, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN_FourierBase(X_left_bd, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN_FourierBase(X_right_bd, Weights_NN1, Biases_NN1, hidden_layers, freq, activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN_FourierBase(X_it, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN_FourierBase(X_left_bd, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN_FourierBase(X_right_bd, Weights_NN2, Biases_NN2, hidden_layers, freq, activate_name=act_func2)
            elif R['model'] == 'PDE_CPDNN':
                # 这个还需要研究，先放在这里。可分解的傅里叶展开，可以直接使用phase shift model
                # 不可以分解的傅里叶展开怎么处理呢？
                U_NN1 = CPDNN_base.CPS_DNN(X_it, index2freq, Weights0_X, Biases0_X, Weights_COS_X, Biases_COS_X,
                                                 Weights_SIN_X, Biases_SIN_X, activate_name=tf.nn.relu)
                ULeft_NN1 = CPDNN_base.CPS_DNN(X_left_bd, index2freq, Weights0_X, Biases0_X, Weights_COS_X,
                                                  Biases_COS_X, Weights_SIN_X, Biases_SIN_X, activate_name=tf.nn.relu)
                URight_NN1 = CPDNN_base.CPS_DNN(X_right_bd, index2freq, Weights0_X, Biases0_X, Weights_COS_X,
                                                   Biases_COS_X, Weights_SIN_X, Biases_SIN_X, activate_name=tf.nn.relu)

            # 变分形式的loss of interior，训练得到的 U_NN1 是 * 行 1 列, 因为 一个点对(x,y) 得到一个 u 值
            if R['variational_loss'] == 1:
                dU_NN1 = tf.gradients(U_NN1, X_it)      # * 行 1 列
                dU_NN2 = tf.gradients(U_NN2, X_it)  # * 行 1 列
                if R['laplace_type'] == 'general_laplace':
                    laplace_norm2NN1 = tf.reduce_sum(tf.square(dU_NN1), axis=-1)
                    loss_it_NN1 = (1.0 / 2) * tf.reshape(laplace_norm2NN1, shape=[-1, 1]) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN1)

                    laplace_norm2NN2 = tf.reduce_sum(tf.square(dU_NN2), axis=-1)
                    loss_it_NN2 = (1.0 / 2) * tf.reshape(laplace_norm2NN2, shape=[-1, 1]) - \
                                          tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN2)
                elif R['laplace_type'] == 'p_laplace':
                    # a_eps = A_eps(X_it)                          # * 行 1 列
                    a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    laplace_p_pow2NN1 = tf.reduce_sum(a_eps*tf.pow(tf.abs(dU_NN1), p_index), axis=-1)
                    loss_it_NN1 = (1.0 / p_index) * tf.reshape(laplace_p_pow2NN1, shape=[-1, 1]) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN1)

                    laplace_p_pow2NN2 = tf.reduce_sum(a_eps * tf.pow(tf.abs(dU_NN2), p_index), axis=-1)
                    loss_it_NN2 = (1.0 / p_index) * tf.reshape(laplace_p_pow2NN2, shape=[-1, 1]) - \
                                          tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN2)

                Loss_it2NN1 = tf.reduce_mean(loss_it_NN1)*(region_r-region_l)
                Loss_it2NN2 = tf.reduce_mean(loss_it_NN2) * (region_r - region_l)

            U_left = u_left(X_left_bd)
            U_right = u_right(X_right_bd)
            loss_bd_NN1 = tf.square(ULeft_NN1 - U_left) + tf.square(URight_NN1 - U_right)
            loss_bd_NN2 = tf.square(ULeft_NN2 - U_left) + tf.square(URight_NN2 - U_right)
            Loss_bd2NN1 = tf.reduce_mean(loss_bd_NN1)
            Loss_bd2NN2 = tf.reduce_mean(loss_bd_NN2)

            if R['regular_weight_model'] == 'L1':
                regular_WB_NN1 = DNN_base.regular_weights_biases_L1(Weights_NN1, Biases_NN1)    # 正则化权重和偏置 L1正则化
                regular_WB_NN2 = DNN_base.regular_weights_biases_L1(Weights_NN2, Biases_NN2)  # 正则化权重和偏置 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB_NN1 = DNN_base.regular_weights_biases_L2(Weights_NN1, Biases_NN1)    # 正则化权重和偏置 L2正则化
                regular_WB_NN2 = DNN_base.regular_weights_biases_L2(Weights_NN2, Biases_NN2)  # 正则化权重和偏置 L2正则化
            else:
                regular_WB_NN1 = tf.constant(0.0)                                         # 无正则化权重参数
                regular_WB_NN2 = tf.constant(0.0)

            penalty_WB = wb_regular * regular_WB_NN1
            Loss2NN1 = Loss_it2NN1 + bd_penalty * Loss_bd2NN1 + wb_regular * regular_WB_NN1       # 要优化的loss function
            Loss2NN2 = Loss_it2NN2 + bd_penalty * Loss_bd2NN2 + wb_regular * regular_WB_NN2

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_Loss2NN1 = my_optimizer.minimize(Loss2NN1, global_step=global_steps)
            train_Loss2NN2 = my_optimizer.minimize(Loss2NN2, global_step=global_steps)

            # 训练上的真解值和训练结果的误差
            U_true = u_true(X_it)
            train_mse_NN1 = tf.reduce_mean(tf.square(U_true - U_NN1))
            train_rel_NN1 = train_mse_NN1 / tf.reduce_mean(tf.square(U_true))

            train_mse_NN2 = tf.reduce_mean(tf.square(U_true - U_NN2))
            train_rel_NN2 = train_mse_NN2 / tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    # 空列表, 使用 append() 添加元素
    lossIt_all2NN1, lossBD_all2NN1, loss_all2NN1, train_mse_all2NN1, train_rel_all2NN1 = [], [], [], [], []
    lossIt_all2NN2, lossBD_all2NN2, loss_all2NN2, train_mse_all2NN2, train_rel_all2NN2 = [], [], [], [], []
    test_mse_all2NN1, test_rel_all2NN1 = [], []
    test_mse_all2NN2, test_rel_all2NN2 = [], []
    test_epoch = []

    test_batch_size = 1000
    test_x_bach = np.reshape(np.linspace(region_l, region_r, num=test_batch_size), [-1, 1])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            x_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r)
            xl_bd_batch, xr_bd_batch = DNN_data.rand_bd_1D(batchsize_bd, input_dim, region_a=region_l, region_b=region_r)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init
            p_WB = 0.0
            _, loss_it_nn1, loss_bd_nn1, loss_nn1, train_mse_nn1, train_rel_nn1 = sess.run(
                [train_Loss2NN1, Loss_it2NN1, Loss_bd2NN1, Loss2NN1, train_mse_NN1, train_rel_NN1],
                feed_dict={X_it: x_it_batch, X_left_bd: xl_bd_batch, X_right_bd: xr_bd_batch,
                           in_learning_rate: tmp_lr, bd_penalty: temp_penalty_bd})
            lossIt_all2NN1.append(loss_it_nn1)
            lossBD_all2NN1.append(loss_bd_nn1)
            loss_all2NN1.append(loss_nn1)
            train_mse_all2NN1.append(train_mse_nn1)
            train_rel_all2NN1.append(train_rel_nn1)

            _, loss_it_nn2, loss_bd_nn2, loss_nn2, train_mse_nn2, train_rel_nn2 = sess.run(
                [train_Loss2NN2, Loss_it2NN2, Loss_bd2NN2, Loss2NN2, train_mse_NN2, train_rel_NN2],
                feed_dict={X_it: x_it_batch, X_left_bd: xl_bd_batch, X_right_bd: xr_bd_batch,
                           in_learning_rate: tmp_lr, bd_penalty: temp_penalty_bd})

            lossIt_all2NN2.append(loss_it_nn2)
            lossBD_all2NN2.append(loss_bd_nn2)
            loss_all2NN2.append(loss_nn2)
            train_mse_all2NN2.append(train_mse_nn2)
            train_rel_all2NN2.append(train_rel_nn2)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, p_WB, loss_it_nn1, loss_bd_nn1, loss_nn1,
                    train_mse_nn1, train_rel_nn1, log_out=log_fileout_NN1)

                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, p_WB, loss_it_nn2, loss_bd_nn2, loss_nn2,
                    train_mse_nn2, train_rel_nn2, log_out=log_fileout_NN2)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, utest_nn1, utest_nn2 = sess.run(
                    [U_true, U_NN1, U_NN2], feed_dict={X_it: test_x_bach, train_opt: train_option})
                test_mse2nn1 = np.mean(np.square(u_true2test - utest_nn1))
                test_mse_all2NN1.append(test_mse2nn1)
                test_rel2nn1 = test_mse2nn1 / np.mean(np.square(u_true2test))
                test_rel_all2NN1.append(test_rel2nn1)

                test_mse2nn2 = np.mean(np.square(u_true2test - utest_nn2))
                test_mse_all2NN2.append(test_mse2nn2)
                test_rel2nn2 = test_mse2nn2 / np.mean(np.square(u_true2test))
                test_rel_all2NN2.append(test_rel2nn2)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn1, test_rel2nn1, log_out=log_fileout_NN1)
                DNN_tools.print_and_log_test_one_epoch(test_mse2nn2, test_rel2nn2, log_out=log_fileout_NN2)

        # -----------------------  save training results to mat files, then plot them ---------------------------------
        saveData.save_trainLoss2mat_1actFunc(lossIt_all2NN1, lossBD_all2NN1, loss_all2NN1, actName=act_func1,
                                             outPath=R['FolderName'])
        saveData.save_trainLoss2mat_1actFunc(lossIt_all2NN2, lossBD_all2NN2, loss_all2NN2, actName=act_func2,
                                             outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(lossIt_all2NN1, lossIt_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(lossBD_all2NN1, lossBD_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                            yaxis_scale=True)
        plotData.plotTrain_losses_2act_funs(loss_all2NN1, loss_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all2NN1, train_rel_all2NN1, actName=act_func1,
                                        outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all2NN2, train_rel_all2NN2, actName=act_func2,
                                        outPath=R['FolderName'])
        plotData.plotTrain_MSEs_2act_funcs(train_mse_all2NN1, train_mse_all2NN2, mseName1=act_func1, mseName2=act_func2,
                                           seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
        plotData.plotTrain_RELs_2act_funcs(train_rel_all2NN1, train_rel_all2NN2, relName1=act_func1, relName2=act_func2,
                                           seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        # ----------------------  save testing results to mat files, then plot them --------------------------------
        saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(utest_nn1, dataName=act_func1, outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(utest_nn2, dataName=act_func2, outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all2NN1, test_rel_all2NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse_all2NN2, test_rel_all2NN2, actName=act_func2, outPath=R['FolderName'])
        plotData.plot_Test_MSE_REL_2ActFuncs(test_mse_all2NN1, test_rel_all2NN1, test_mse_all2NN2,
                                             test_rel_all2NN2, test_epoch, actName1=act_func1, actName2=act_func2,
                                             seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    store_file = 'laplace1d'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # R['laplace_type'] = 'general_laplace'
    # R['eqs_name'] = 'PDE1'
    # R['eqs_name'] = 'PDE2'
    # R['eqs_name'] = 'PDE3'
    # R['eqs_name'] = 'PDE4'
    # R['eqs_name'] = 'PDE5'
    # R['eqs_name'] = 'PDE6'
    # R['eqs_name'] = 'PDE7'

    R['laplace_type'] = 'p_laplace'
    R['eqs_name'] = 'multi_scale'

    if R['laplace_type'] == 'general_laplace':
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
    elif R['laplace_type'] == 'p_laplace':
        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)              # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2laplace'] = order

    R['input_dim'] = 1                         # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                        # 输出维数
    R['variational_loss'] = 1                  # PDE变分

    # ---------------------------- Setup of DNN -------------------------------
    R['batch_size2interior'] = 3000            # 内部训练数据的批大小
    R['batch_size2boundary'] = 500             # 边界训练数据大小

    R['weight_biases_model'] = 'general_model'
    # R['weight_biases_model'] = 'phase_shift_model'

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    R['regular_weight_biases'] = 0.000     # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001   # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025  # Regularization parameter for weights

    R['activate_stage_penalty'] = 1
    R['boundary_penalty'] = 100                           # Regularization parameter for boundary conditions

    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['optimizer_name'] = 'Adam'                          # 优化器

    # R['hidden_layers'] = (80, 60, 40, 20, 10)
    # R['hidden_layers'] = (200, 100, 80, 50, 30)
    R['hidden_layers'] = (300, 200, 150, 150, 100, 50, 50)
    # R['hidden_layers'] = (400, 300, 300, 200, 100, 100, 50)
    # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
    # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)
    # R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100)
    # R['hidden_layers'] = (500, 300, 200, 200, 100, 100, 50)

    if R['epsilon'] == 0.01:
        R['hidden_layers'] = (400, 300, 300, 200, 100, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100)

    # R['model'] = 'PDE_DNN'                           # 使用的网络模型
    # R['model'] = 'PDE_DNN_BN'
    # R['model'] = 'PDE_DNN_scale'
    # R['model'] = 'PDE_DNN_adapt_scale'
    R['model'] = 'PDE_DNN_FourierBase'
    # R['model'] = 'PDE_CPDNN'

    # 激活函数的选择
    # R['act_name2NN1'] = 'relu'
    R['act_name2NN1'] = 'tanh'
    # R['act_name2NN1'] = 'srelu'
    # R['act_name2NN1'] = 'sin'
    # R['act_name2NN1'] = 's2relu'

    # R['act_name2NN2'] = 'relu'
    # R['act_name2NN2']' = leaky_relu'
    # R['act_name2NN2'] = 'srelu'
    R['act_name2NN2'] = 's2relu'
    # R['act_name2NN2'] = 'powsin_srelu'
    # R['act_name2NN2'] = 'slrelu'
    # R['act_name2NN2'] = 'elu'
    # R['act_name2NN2'] = 'selu'
    # R['act_name2NN2'] = 'phi'

    R['plot_ongoing'] = 0
    R['subfig_type'] = 0

    solve_laplace(R)

