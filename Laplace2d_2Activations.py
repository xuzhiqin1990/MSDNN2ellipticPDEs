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
import time
import DNN_base
import DNN_tools
import DNN_data
import MS_Laplace_eqs
import general_laplace_eqs
import matData2multi_scale
import saveData
import plotData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout, actName=None):
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    DNN_tools.log_string('The order to p-laplacian: %s\n' % (R_dic['order2laplace']), log_fileout)

    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(actName), log_fileout)

    if R['PDE_type'] == 'p_laplace2multi_scale_implicit' or R['PDE_type'] == 'p_laplace2multi_scale_explicit':
        DNN_tools.log_string('epsilon: %f\n' % (R_dic['epsilon']), log_fileout)  # 替换上两行

    if R['PDE_type'] == 'p_laplace2multi_scale_implicit':
        DNN_tools.log_string('The mesh_number: %f\n' % (R['mesh_number']), log_fileout)  # 替换上两行

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

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['boundary_penalty']                # Regularization parameter for boundary conditions
    wb_regular = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']
    act_func1 = R['act_name2NN1']
    act_func2 = R['act_name2NN2']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # p laplace 问题需要的额外设置, 先预设一下
    p = 2
    epsilon = 0.1
    mesh_number = 2

    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'general_laplace':
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u_left, u_right, u_bottom, u_top = general_laplace_eqs.get_general_laplace_infos(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, laplace_name=R['equa_name'])
    elif R['PDE_type'] == 'p_laplace2multi_scale_implicit':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
        p = R['order2laplace']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'multi_scale2D_5':
            region_lb = 0.0
            region_rt = 1.0
        else:
            region_lb = -1.0
            region_rt = 1.0
        u_true, f, A_eps, u_left, u_right, u_bottom, u_top = MS_Laplace_eqs.get_laplace_multi_scale_infos(
                input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], region_lb=0.0, region_rt=1.0,
                laplace_name=R['equa_name'])
    elif R['PDE_type'] == 'p_laplace2multi_scale_explicit':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        if R['equa_name'] == 'multi_scale2D_6':
            region_lb = -1.0
            region_rt = 1.0
            f = MS_Laplace_eqs.force_side2E6(input_dim, out_dim)                       # f是一个向量
            u_true = MS_Laplace_eqs.true_solution2E6(input_dim, out_dim, eps=epsilon)
            u_left, u_right, u_bottom, u_top = MS_Laplace_eqs.boundary2E6(input_dim, out_dim, region_lb, region_rt,
                                                                          eps=epsilon)
            # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
            A_eps = MS_Laplace_eqs.elliptic_coef2E6(input_dim, out_dim, eps=epsilon)

    flag2NN1 = 'WB2NN1'
    flag2NN2 = 'WB_NN2'
    # Weights, Biases = DNN_base.Initial_DNN2different_hidden(input_dim, out_dim, hidden_layers, flag)
    # Weights, Biases = DNN_base.initialize_NN_xavier(input_dim, out_dim, hidden_layers, flag)
    # Weights, Biases = DNN_base.initialize_NN_random_normal(input_dim, out_dim, hidden_layers, flag)
    W2NN1, B2NN1 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2NN1)
    W2NN2, B2NN2 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2NN2)

    if R['model'] == 'laplace_DNN_adapt_scale':
        freq_frag = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
        Unit_num = int(hidden_layers[0] / len(freq_frag))
        # np.repeat(a, repeats, axis=None)
        # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
        # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
        init_mixcoe = np.repeat(freq_frag, Unit_num)
        init_mixcoe = np.concatenate((init_mixcoe, np.ones([hidden_layers[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
        init_mixcoe = init_mixcoe.astype(np.float32)
    else:
        init_mixcoe = np.array([[0.0]])
        init_mixcoe = init_mixcoe.astype(np.float32)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XY_it = tf.placeholder(tf.float32, name='X_it', shape=[None, input_dim])                # * 行 2 列
            XY_left_bd = tf.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])      # * 行 2 列
            XY_right_bd = tf.placeholder(tf.float32, name='X_right_bd', shape=[None, input_dim])    # * 行 2 列
            XY_bottom_bd = tf.placeholder(tf.float32, name='Y_bottom_bd', shape=[None, input_dim])  # * 行 2 列
            XY_top_bd = tf.placeholder(tf.float32, name='Y_top_bd', shape=[None, input_dim])        # * 行 2 列
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')
            multi_scale_ceof = tf.Variable(initial_value=init_mixcoe, dtype='float32', name='M0')

            # 供选择的网络模式
            if R['model'] == 'PDE_DNN':
                U_NN1 = DNN_base.PDE_DNN(XY_it, W2NN1, B2NN1, hidden_layers, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN(XY_left_bd, W2NN1, B2NN1, hidden_layers, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN(XY_right_bd, W2NN1, B2NN1, hidden_layers, activate_name=act_func1)
                UBottom_NN1 = DNN_base.PDE_DNN(XY_bottom_bd, W2NN1, B2NN1, hidden_layers, activate_name=act_func1)
                UTop_NN1 = DNN_base.PDE_DNN(XY_top_bd, W2NN1, B2NN1, hidden_layers, activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN(XY_it, W2NN2, B2NN2, hidden_layers, activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN(XY_left_bd, W2NN2, B2NN2, hidden_layers, activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN(XY_right_bd, W2NN2, B2NN2, hidden_layers, activate_name=act_func2)
                UBottom_NN2 = DNN_base.PDE_DNN(XY_bottom_bd, W2NN2, B2NN2, hidden_layers, activate_name=act_func2)
                UTop_NN2 = DNN_base.PDE_DNN(XY_top_bd, W2NN2, B2NN2, hidden_layers, activate_name=act_func2)
            elif R['model'] == 'PDE_DNN_scale':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN1 = DNN_base.PDE_DNN_scale(XY_it, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN_scale(XY_left_bd, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN_scale(XY_right_bd, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)
                UBottom_NN1 = DNN_base.PDE_DNN_scale(XY_bottom_bd, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)
                UTop_NN1 = DNN_base.PDE_DNN_scale(XY_top_bd, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)

                U_NN2= DNN_base.PDE_DNN_scale(XY_it, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
                ULeft_NN2= DNN_base.PDE_DNN_scale(XY_left_bd, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
                URight_NN2= DNN_base.PDE_DNN_scale(XY_right_bd, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
                UBottom_NN2= DNN_base.PDE_DNN_scale(XY_bottom_bd, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
                UTop_NN2= DNN_base.PDE_DNN_scale(XY_top_bd, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
            elif R['model'] == 'PDE_DNN_FourierBase':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN1 = DNN_base.PDE_DNN_FourierBase(XY_it, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN_FourierBase(XY_left_bd, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN_FourierBase(XY_right_bd, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)
                UBottom_NN1 = DNN_base.PDE_DNN_FourierBase(XY_bottom_bd, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)
                UTop_NN1 = DNN_base.PDE_DNN_FourierBase(XY_top_bd, W2NN1, B2NN1, hidden_layers, freq, activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN_FourierBase(XY_it, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN_FourierBase(XY_left_bd, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN_FourierBase(XY_right_bd, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
                UBottom_NN2 = DNN_base.PDE_DNN_FourierBase(XY_bottom_bd, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
                UTop_NN2 = DNN_base.PDE_DNN_FourierBase(XY_top_bd, W2NN2, B2NN2, hidden_layers, freq, activate_name=act_func2)
            elif R['model'] == 'PDE_DNN_adapt_scale':
                freqs = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN1 = DNN_base.PDE_DNN_adapt_scale(XY_it, W2NN1, B2NN1, hidden_layers, freqs, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN_adapt_scale(XY_left_bd, W2NN1, B2NN1, hidden_layers, freqs, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN_adapt_scale(XY_right_bd, W2NN1, B2NN1, hidden_layers, freqs, activate_name=act_func1)
                UBottom_NN1 = DNN_base.PDE_DNN_adapt_scale(XY_bottom_bd, W2NN1, B2NN1, hidden_layers, freqs, activate_name=act_func1)
                UTop_NN1 = DNN_base.PDE_DNN_adapt_scale(XY_top_bd, W2NN1, B2NN1, hidden_layers, freqs, activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN_adapt_scale(XY_it, W2NN2, B2NN2, hidden_layers, freqs, activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN_adapt_scale(XY_left_bd, W2NN2, B2NN2, hidden_layers, freqs, activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN_adapt_scale(XY_right_bd, W2NN2, B2NN2, hidden_layers, freqs, activate_name=act_func2)
                UBottom_NN2 = DNN_base.PDE_DNN_adapt_scale(XY_bottom_bd, W2NN2, B2NN2, hidden_layers, freqs, activate_name=act_func2)
                UTop_NN2 = DNN_base.PDE_DNN_adapt_scale(XY_top_bd, W2NN2, B2NN2, hidden_layers, freqs, activate_name=act_func2)

            X_it = tf.reshape(XY_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XY_it[:, 1], shape=[-1, 1])
            if R['variational_loss'] == 1:
                dU_NN1 = tf.gradients(U_NN1, XY_it)[0]      # * 行 2 列
                dU_hat_norm2sin = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dU_NN1), axis=-1)), shape=[-1, 1])  # 按行求和

                dU_NN2 = tf.gradients(U_NN2, XY_it)[0]  # * 行 2 列
                dU_hat_norm2srelu = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dU_NN2), axis=-1)), shape=[-1, 1])  # 按行求和
                if R['PDE_type'] == 'general_laplace':
                    laplace_pow2sin = tf.square(dU_hat_norm2sin)
                    loss_it_variational2sin = (1.0 / 2) *laplace_pow2sin - tf.multiply(f(X_it, Y_it), U_NN1)

                    laplace_pow2srelu = tf.square(dU_hat_norm2srelu)
                    loss_it_variational2srelu = (1.0 / 2) * laplace_pow2srelu - tf.multiply(f(X_it, Y_it), U_NN2)
                else:
                    a_eps = A_eps(X_it, Y_it)                          # * 行 1 列

                    laplace_p_pow2sin = a_eps*tf.pow(dU_hat_norm2sin, p)
                    loss_it_variational2sin = (1.0 / p) * laplace_p_pow2sin - tf.multiply(f(X_it, Y_it), U_NN1)

                    laplace_p_pow2srelu = a_eps * tf.pow(dU_hat_norm2srelu, p)
                    loss_it_variational2srelu = (1.0 / p) * laplace_p_pow2srelu - tf.multiply(f(X_it, Y_it), U_NN2)
                Loss_it2NN1 = tf.reduce_mean(loss_it_variational2sin)*(region_rt-region_lb)*(region_rt-region_lb)
                Loss_it2NN2 = tf.reduce_mean(loss_it_variational2srelu) * (region_rt - region_lb) * (
                            region_rt - region_lb)

            if 1 == R['input_dim']:
                U_left = u_left(tf.reshape(XY_left_bd[:, 0], shape=[-1, 1]),
                                tf.reshape(XY_left_bd[:, 1], shape=[-1, 1]))
                U_right = u_right(tf.reshape(XY_right_bd[:, 0], shape=[-1, 1]),
                                  tf.reshape(XY_right_bd[:, 1], shape=[-1, 1]))
                loss_bd_square2NN1 = tf.square(ULeft_NN1 - U_left) + tf.square(URight_NN1 - U_right)
                Loss_bd2NN1 = tf.reduce_mean(loss_bd_square2NN1)

                loss_bd_square2NN2 = tf.square(ULeft_NN2 - U_left) + tf.square(URight_NN2 - U_right)
                Loss_bd2NN2 = tf.reduce_mean(loss_bd_square2NN2)
            elif 2 == R['input_dim']:
                U_left = u_left(tf.reshape(XY_left_bd[:, 0], shape=[-1, 1]), tf.reshape(XY_left_bd[:, 1], shape=[-1, 1]))
                U_right = u_right(tf.reshape(XY_right_bd[:, 0], shape=[-1, 1]), tf.reshape(XY_right_bd[:, 1], shape=[-1, 1]))
                U_bottom = u_bottom(tf.reshape(XY_bottom_bd[:, 0], shape=[-1, 1]), tf.reshape(XY_bottom_bd[:, 1], shape=[-1, 1]))
                U_top = u_top(tf.reshape(XY_top_bd[:, 0], shape=[-1, 1]), tf.reshape(XY_top_bd[:, 1], shape=[-1, 1]))

                loss_bd_square2NN1 = tf.square(ULeft_NN1 - U_left) + tf.square(URight_NN1 - U_right) + \
                                 tf.square(UBottom_NN1 - U_bottom) + tf.square(UTop_NN1 - U_top)
                Loss_bd2NN1 = tf.reduce_mean(loss_bd_square2NN1)

                loss_bd_square2NN2 = tf.square(ULeft_NN2 - U_left) + tf.square(URight_NN2 - U_right) + \
                                 tf.square(UBottom_NN2 - U_bottom) + tf.square(UTop_NN2 - U_top)
                Loss_bd2NN2= tf.reduce_mean(loss_bd_square2NN2)

            if R['regular_weight_model'] == 'L1':
                regular_WB2sin = DNN_base.regular_weights_biases_L1(W2NN1, B2NN1)    # 正则化权重和偏置 L1正则化
                regular_WB2srelu = DNN_base.regular_weights_biases_L1(W2NN2, B2NN2)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2sin = DNN_base.regular_weights_biases_L2(W2NN1, B2NN1)    # 正则化权重和偏置 L2正则化
                regular_WB2srelu = DNN_base.regular_weights_biases_L2(W2NN2, B2NN2)
            else:
                regular_WB2sin = 0.0                                         # 无正则化权重参数
                regular_WB2srelu = 0.0

            Loss2NN1 = Loss_it2NN1 + bd_penalty * Loss_bd2NN1 + wb_regular * regular_WB2sin

            Loss2NN2 = Loss_it2NN2 + bd_penalty * Loss_bd2NN2 + wb_regular * regular_WB2srelu

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['train_group'] == 1:
                train_sin_op1 = my_optimizer.minimize(Loss_it2NN1, global_step=global_steps)
                train_sin_op2 = my_optimizer.minimize(Loss_bd2NN1, global_step=global_steps)
                train_sin_op3 = my_optimizer.minimize(Loss2NN1, global_step=global_steps)
                train_loss2sin = tf.group(train_sin_op1, train_sin_op2, train_sin_op3)

                train_srelu_op1 = my_optimizer.minimize(Loss_it2NN2, global_step=global_steps)
                train_srelu_op2 = my_optimizer.minimize(Loss_bd2NN2, global_step=global_steps)
                train_srelu_op3 = my_optimizer.minimize(Loss2NN2, global_step=global_steps)
                train_loss2srelu = tf.group(train_srelu_op1, train_srelu_op2, train_srelu_op3)
            elif R['train_group'] == 2:
                train_sin_op1 = my_optimizer.minimize(Loss2NN1, global_step=global_steps)
                train_sin_op2 = my_optimizer.minimize(Loss_bd2NN1, global_step=global_steps)
                train_loss2sin = tf.group(train_sin_op1, train_sin_op2)

                train_srelu_op2 = my_optimizer.minimize(Loss2NN2, global_step=global_steps)
                train_srelu_op1 = my_optimizer.minimize(Loss_bd2NN2, global_step=global_steps)
                train_loss2srelu = tf.group(train_srelu_op1, train_srelu_op2)
            else:
                train_loss2sin = my_optimizer.minimize(Loss2NN1, global_step=global_steps)
                train_loss2srelu = my_optimizer.minimize(Loss2NN2, global_step=global_steps)

            if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'p_laplace2multi_scale_explicit':
                # 训练上的真解值和训练结果的误差
                Utrue = u_true(X_it, Y_it)

                train_mse2NN1 = tf.reduce_mean(tf.square(Utrue - U_NN1))
                train_rel2NN1 = train_mse2NN1 / tf.reduce_mean(tf.square(Utrue))

                train_mse2NN2 = tf.reduce_mean(tf.square(Utrue - U_NN2))
                train_rel2NN2 = train_mse2NN2 / tf.reduce_mean(tf.square(Utrue))
            else:
                train_mse2NN1 = tf.constant(0.0)
                train_rel2NN1 = tf.constant(0.0)

                train_mse2NN2 = tf.constant(0.0)
                train_rel2NN2 = tf.constant(0.0)

    t0 = time.time()
    # 空列表, 使用 append() 添加元素
    lossIt_all2NN1, lossBD_all2NN1, loss_all2NN1, train_mse_all2NN1, train_rel_all2NN1 = [], [], [], [], []
    lossIt_all2NN2, lossBD_all2NN2, loss_all2NN2, train_mse_all2NN2, train_rel_all2NN2 = [], [], [], [], []
    test_mse_all2NN1, test_rel_all2NN1 = [], []
    test_mse_all2NN2, test_rel_all2NN2 = [], []
    test_epoch = []

    # 画网格热力解图 ---- 生成测试数据，用于测试训练后的网络
    if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'p_laplace2multi_scale_explicit':
        # test_bach_size = 400
        # size2test = 20
        test_bach_size = 900
        size2test = 30
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        # test_bach_size = 40000
        # size2test = 200
        # test_bach_size = 250000
        # size2test = 500
        # test_bach_size = 1000000
        # size2test = 1000
        test_xy_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        # test_x_bach = np.reshape(test_xy_bach[:, 0], newshape=[-1, 1])
        # test_y_bach = np.reshape(test_xy_bach[:, 1], newshape=[-1, 1])
    elif R['PDE_type'] == 'p_laplace2multi_scale_implicit':
        test_xy_bach = matData2multi_scale.get_data2multi_scale(equation_name=R['equa_name'], mesh_number=mesh_number)
        # test_x_bach = test_xy_bach[:, 0]
        # test_y_bach = test_xy_bach[:, 1]
        size2batch = np.shape(test_xy_bach)[0]
        size2test = int(np.sqrt(size2batch))

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        train_option = True
        for i_epoch in range(R['max_epoch'] + 1):
            xy_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xl_bd_batch, xr_bd_batch, yb_bd_batch, yt_bd_batch = \
                DNN_data.rand_bd_2D(batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_penalty2bd_increase'] == 1:
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
            elif R['activate_penalty2bd_increase'] == 2:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = 5*bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 1 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 0.5 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 0.1 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 0.05 * bd_penalty_init
                else:
                    temp_penalty_bd = 0.02 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            _, loss_it_nn1, loss_bd_nn1, loss_nn1, train_mse_nn1, train_rel_nn1 = sess.run(
                [train_loss2sin, Loss_it2NN1, Loss_bd2NN1, Loss2NN1, train_mse2NN1, train_rel2NN1],
                feed_dict={XY_it: xy_it_batch, XY_left_bd: xl_bd_batch, XY_right_bd: xr_bd_batch,
                           XY_bottom_bd: yb_bd_batch, XY_top_bd: yt_bd_batch, in_learning_rate: tmp_lr,
                           bd_penalty: temp_penalty_bd, train_opt: train_option})
            lossIt_all2NN1.append(loss_it_nn1)
            lossBD_all2NN1.append(loss_bd_nn1)
            loss_all2NN1.append(loss_nn1)
            train_mse_all2NN1.append(train_mse_nn1)
            train_rel_all2NN1.append(train_rel_nn1)

            _, loss_it_nn2, loss_bd_nn2, loss_nn2, train_mse_nn2, train_rel_nn2 = sess.run(
                [train_loss2srelu, Loss_it2NN2, Loss_bd2NN2, Loss2NN2, train_mse2NN2, train_rel2NN2],
                feed_dict={XY_it: xy_it_batch, XY_left_bd: xl_bd_batch, XY_right_bd: xr_bd_batch,
                                    XY_bottom_bd: yb_bd_batch, XY_top_bd: yt_bd_batch, in_learning_rate: tmp_lr,
                                    bd_penalty: temp_penalty_bd, train_opt: train_option})

            lossIt_all2NN2.append(loss_it_nn2)
            lossBD_all2NN2.append(loss_bd_nn2)
            loss_all2NN2.append(loss_nn2)
            train_mse_all2NN2.append(train_mse_nn2)
            train_rel_all2NN2.append(train_rel_nn2)

            if i_epoch % 1000 == 0:
                p_WB = 0.0
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
                if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'p_laplace2multi_scale_explicit':
                    u_true2test, utest_nn1, utest_nn2 = sess.run(
                        [Utrue, U_NN1, U_NN2], feed_dict={XY_it: test_xy_bach, train_opt: train_option})
                else:
                    u_true2test = u_true
                    utest_nn1, utest_nn2 = sess.run([U_NN1, U_NN2],
                                                        feed_dict={XY_it: test_xy_bach, train_opt: train_option})
                point_ERR2NN1 = np.square(u_true2test - utest_nn1)
                test_mse2nn1 = np.mean(point_ERR2NN1)
                test_mse_all2NN1.append(test_mse2nn1)
                test_rel2nn1 = test_mse2nn1 / np.mean(np.square(u_true2test))
                test_rel_all2NN1.append(test_rel2nn1)

                point_ERR2NN2 = np.square(u_true2test - utest_nn2)
                test_mse2nn2 = np.mean(point_ERR2NN2)
                test_mse_all2NN2.append(test_mse2nn2)
                test_rel2nn2 = test_mse2nn2 / np.mean(np.square(u_true2test))
                test_rel_all2NN2.append(test_rel2nn2)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn1, test_rel2nn1, log_out=log_fileout_NN1)
                DNN_tools.print_and_log_test_one_epoch(test_mse2nn2, test_rel2nn2, log_out=log_fileout_NN2)

        # ------------------- save the testing results into mat file and plot them -------------------------
        saveData.save_trainLoss2mat_1actFunc(lossIt_all2NN1, lossBD_all2NN1, loss_all2NN1, actName=act_func1,
                                             outPath=R['FolderName'])
        saveData.save_trainLoss2mat_1actFunc(lossIt_all2NN2, lossBD_all2NN2, loss_all2NN2, actName=act_func2,
                                             outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all2NN1, train_rel_all2NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all2NN2, train_rel_all2NN2, actName=act_func2, outPath=R['FolderName'])

        plotData.plotTrain_losses_2act_funs(lossIt_all2NN1, lossIt_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(lossBD_all2NN1, lossBD_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
        plotData.plotTrain_losses_2act_funs(loss_all2NN1, loss_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plotTrain_MSEs_2act_funcs(train_mse_all2NN1, train_mse_all2NN2, mseName1=act_func1, mseName2=act_func2,
                                           seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        plotData.plotTrain_RELs_2act_funcs(train_rel_all2NN1, train_rel_all2NN2, relName1=act_func1, relName2=act_func2,
                                           seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        # ----------------- save test data to mat file and plot the testing results into figures -----------------------
        if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'p_laplace2multi_scale_explicit':
            saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])

        saveData.save_testData_or_solus2mat(utest_nn1, dataName=act_func1, outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(utest_nn2, dataName=act_func2, outPath=R['FolderName'])

        if R['hot_power'] == 0:
            # ----------------------------------------------------------------------------------------------------------
            #                                      绘制解的3D散点图(真解和DNN解)
            # ----------------------------------------------------------------------------------------------------------
            plotData.plot_scatter_solutions2test(u_true2test, utest_nn1, test_xy_bach, actName1='Utrue',
                                                 actName2=act_func1, seedNo=R['seed'], outPath=R['FolderName'])
            plotData.plot_scatter_solutions2test(u_true2test, utest_nn2, test_xy_bach, actName1='Utrue',
                                                 actName2=act_func2, seedNo=R['seed'], outPath=R['FolderName'])
        elif R['hot_power'] == 1:
            # ----------------------------------------------------------------------------------------------------------
            #                                      绘制解的热力图(真解和DNN解)
            # ----------------------------------------------------------------------------------------------------------
            plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue', seedNo=R['seed'],
                                            outPath=R['FolderName'])
            plotData.plot_Hot_solution2test(utest_nn1, size_vec2mat=size2test, actName=act_func1, seedNo=R['seed'],
                                            outPath=R['FolderName'])
            plotData.plot_Hot_solution2test(utest_nn2, size_vec2mat=size2test, actName=act_func2, seedNo=R['seed'],
                                            outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all2NN1, test_rel_all2NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse_all2NN2, test_rel_all2NN2, actName=act_func2, outPath=R['FolderName'])

        plotData.plot_2TestMSEs(test_mse_all2NN1, test_mse_all2NN2, mseType1='s2ReLU', mseType2='sReLU',
                                epoches=test_epoch, seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_2TestRELs(test_rel_all2NN1, test_rel_all2NN2, relType1='s2ReLU', relType2='sReLU',
                                epoches=test_epoch, seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plot_Test_MSE_REL_2ActFuncs(test_mse_all2NN1, test_rel_all2NN1, test_mse_all2NN2, test_rel_all2NN2,
                                             epoches=test_epoch, actName1=act_func1, actName2=act_func2, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        saveData.save_test_point_wise_err2mat(point_ERR2NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_test_point_wise_err2mat(point_ERR2NN2, actName=act_func2, outPath=R['FolderName'])

        plotData.plot_Hot_point_wise_err(point_ERR2NN1, size_vec2mat=size2test, actName=act_func1,
                                         seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_Hot_point_wise_err(point_ERR2NN2, size_vec2mat=size2test, actName=act_func2,
                                         seedNo=R['seed'], outPath=R['FolderName'])


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

    # 文件保存路径设置
    store_file = 'laplace2d'
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

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    R['input_dim'] = 2                # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1               # 输出维数

    # ---------------------------- Setup of multi-scale problem-------------------------------
    # R['PDE_type'] = 'general_laplace'
    # R['equa_name'] = 'PDE1'
    # R['equa_name'] = 'PDE2'
    # R['equa_name'] = 'PDE3'
    # R['equa_name'] = 'PDE4'
    # R['equa_name'] = 'PDE5'
    # R['equa_name'] = 'PDE6'
    # R['equa_name'] = 'PDE7'

    R['PDE_type'] = 'p_laplace2multi_scale_implicit'
    # R['equa_name'] = 'multi_scale2D_1'
    R['equa_name'] = 'multi_scale2D_2'
    # R['equa_name'] = 'multi_scale2D_3'
    # R['equa_name'] = 'multi_scale2D_4'
    # R['equa_name'] = 'multi_scale2D_4'
    # R['equa_name'] = 'multi_scale2D_5'    # p=3

    # R['PDE_type'] = 'p_laplace2multi_scale_explicit'
    # R['equa_name'] = 'multi_scale2D_6'

    if R['PDE_type'] == 'general_laplace':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
        R['batch_size2interior'] = 3000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 500
    elif R['PDE_type'] == 'p_laplace2multi_scale_implicit':
        # 频率设置
        mesh_number = input('please input mesh_number =')  # 由终端输入的会记录为字符串形式
        R['mesh_number'] = int(mesh_number)  # 字符串转为浮点

        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)  # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2laplace'] = order

        R['batch_size2interior'] = 3000  # 内部训练数据的批大小
        if R['mesh_number'] == 2:
            R['batch_size2boundary'] = 25  # 边界训练数据的批大小
        elif R['mesh_number'] == 3:
            R['batch_size2boundary'] = 100  # 边界训练数据的批大小
        elif R['mesh_number'] == 4:
            R['batch_size2boundary'] = 200  # 边界训练数据的批大小
        elif R['mesh_number'] == 5:
            R['batch_size2boundary'] = 300  # 边界训练数据的批大小
        elif R['mesh_number'] == 6:
            R['batch_size2boundary'] = 500  # 边界训练数据的批大小
    elif R['PDE_type'] == 'p_laplace2multi_scale_explicit':
        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)  # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2laplace'] = order
        R['batch_size2interior'] = 3000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 500  # 边界训练数据的批大小

    # ---------------------------- Setup of DNN -------------------------------
    R['weight_biases_model'] = 'general_model'

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'

    R['regular_weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025                 # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['boundary_penalty'] = 1000                          # Regularization parameter for boundary conditions

    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['optimizer_name'] = 'Adam'                          # 优化器
    R['train_group'] = 0

    R['hidden_layers'] = (12, 10, 8, 8, 6)
    # R['hidden_layers'] = (100, 80, 60, 60, 40, 40, 20)
    # R['hidden_layers'] = (200, 100, 80, 50, 30)
    # R['hidden_layers'] = (300, 200, 150, 100, 100, 50, 50)
    # R['hidden_layers'] = (500, 400, 300, 200, 100)
    # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
    # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)
    # R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100)
    # R['hidden_layers'] = (500, 300, 200, 200, 100, 100, 50)
    # R['hidden_layers'] = (1000, 800, 600, 400, 200)
    # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)
    # R['hidden_layers'] = (2000, 1500, 1000, 500, 250)

    # R['model'] = 'PDE_DNN'                         # 使用的网络模型
    # R['model'] = 'PDE_DNN_BN'
    R['model'] = 'PDE_DNN_scale'
    # R['model'] = 'PDE_DNN_adapt_scale'
    # R['model'] = 'PDE_DNN_FourierBase'
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

    R['variational_loss'] = 1                            # PDE变分
    R['hot_power'] = 1
    solve_laplace(R)

