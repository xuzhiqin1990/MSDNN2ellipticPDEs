# -*- coding: utf-8 -*-
"""
Modify on 2020年4月1日

@author: LXG
Benchmark Code of Coupled PhaseDNN for ODE. 
"""

import tensorflow as tf
import numpy as np


# ---------------------------------------------- my activations -----------------------------------------------
def srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)


def sin_srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)


def sin2_srelu(x):
    return 2.0*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(4*np.pi*x)*tf.sin(2*np.pi*x)


def slrelu(x):
    return tf.nn.leaky_relu(1-x)*tf.nn.leaky_relu(x)


def pow2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.nn.relu(x)


def selu(x):
    return tf.nn.elu(1-x)*tf.nn.elu(x)


def wave(x):
    return tf.nn.relu(x) - 2*tf.nn.relu(x-1/4) + \
           2*tf.nn.relu(x-3/4) - tf.nn.relu(x-1)


def phi(x):
    return tf.nn.relu(x) * tf.nn.relu(x)-3*tf.nn.relu(x-1)*tf.nn.relu(x-1) + 3*tf.nn.relu(x-2)*tf.nn.relu(x-2) \
           - tf.nn.relu(x-3)*tf.nn.relu(x-3)*tf.nn.relu(x-3)


# 生成DNN的权重和偏置
# layers indicates the number of HIDDEN LAYERS, without input and output layers
# tf.random_normal(): 用于从服从指定正太分布的数值中取出随机数
# tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
# hape: 输出张量的形状，必选.--- mean: 正态分布的均值，默认为0.----stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32 ----seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样---name: 操作的名称
def Initial_DNN(in_size, out_size, hidden_index, variateFlag):
    layers = len(hidden_index)
    Weights = []  # 权重列表，用于存储隐藏层的权重
    Biases = []  # 偏置列表，用于存储隐藏层的偏置
    # 第一层的权重和偏置，对输入数据做变换
    W = tf.Variable(0.1 * tf.random.normal([in_size, hidden_index[0]]), dtype='float32',
                    name='W-transInput' + str(variateFlag))
    B = tf.Variable(0.1 * tf.random.uniform([1, hidden_index[0]]), dtype='float32',
                    name='B-transInput' + str(variateFlag))
    Weights.append(W)
    Biases.append(B)

    # 隐藏层：第二至倒数第二层的权重和偏置
    for i_layer in range(layers - 1):
        W = tf.Variable(0.1 * tf.random.normal([hidden_index[i_layer], hidden_index[i_layer+1]]), dtype='float32',
                        name='W-hidden' + str(i_layer + 1) + str(variateFlag))
        B = tf.Variable(0.1 * tf.random.uniform([1, hidden_index[i_layer+1]]), dtype='float32',
                        name='B-hidden' + str(i_layer + 1) + str(variateFlag))
        Weights.append(W)
        Biases.append(B)

    # 最后一层的权重和偏置。将最后的结果变换到输出维度
    W = tf.Variable(0.1 * tf.random.normal([hidden_index[-1], out_size]), dtype='float32',
                    name='W-outTrans' + str(variateFlag))
    B = tf.Variable(0.1 * tf.random.uniform([1, out_size]), dtype='float32',
                    name='B-outTrans' + str(variateFlag))
    Weights.append(W)
    Biases.append(B)

    return Weights, Biases


# 这里可不可以利用attention 呢？关联 input 和 output。得到 attention 系数后，然后作用到input上
# 但是如何让input 和 output attention 关联起来呢 ？
def multilayer_DNN_residual(variable_input, Weights, Biases, CPDNN_activation=tf.nn.relu):
    layers = len(Weights)                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    beta = tf.constant(0.1)
    H_pre = variable_input
    W_pre = Weights[0]
    dims_pre = tf.shape(W_pre)
    for k in range(layers-1):
        W = Weights[k]
        B = Biases[k]
        H = CPDNN_activation(tf.add(tf.matmul(H, W), B)) * CPDNN_activation(1 - tf.add(tf.matmul(H, W), B))

        dim_post = tf.shape(W)
        if dim_post[-1] == dims_pre[-1]:
            H = H + beta*H_pre

        H_pre = H
        W_pre = W
        dims_pre = tf.shape(W_pre)

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # dim_post = tf.shape(W_out)
    # if dim_post[-1] == dims_pre[-1]:
    #     output = output + beta * H
    output = tf.nn.tanh(output)
    return output


# 这里可不可以利用attention 呢？关联 input 和 output。得到 attention 系数后，然后作用到input上
# 但是如何让input 和 output attention 关联起来呢 ？
def multilayer_DNN_attention(variable_input, Weights, Biases, CPDNN_activation=tf.nn.relu):
    layers = len(Weights)                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    for k in range(layers-1):
        W = Weights[k]
        B = Biases[k]
        # H = CPDNN_activation(tf.add(tf.matmul(H, W), B))
        H = CPDNN_activation(tf.add(tf.matmul(H, W), B)) * CPDNN_activation(1 - tf.add(tf.matmul(H, W), B))
    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    output = tf.nn.tanh(output)
    return output


def multilayer_DNN(variable_input, Weights, Biases, CPDNN_activation=tf.nn.relu):
    layers = len(Weights)                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    for k in range(layers-1):
        W = Weights[k]
        B = Biases[k]
        # H = CPDNN_activation(tf.add(tf.matmul(H, W), B))
        H = CPDNN_activation(tf.add(tf.matmul(H, W), B)) * CPDNN_activation(1 - tf.add(tf.matmul(H, W), B))
    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    output = tf.nn.tanh(output)
    return output


def CPS_DNN(input_x, freqs, Weights0, Biases0, Weights_COS, Biases_COS, Weights_SIN, Biases_SIN,
            activate_name=tf.nn.relu):
    if activate_name == 'relu':
        CPDNN_activation = tf.nn.relu
    elif activate_name == 'leaky_relu':
        CPDNN_activation = tf.nn.leaky_relu(0.2)
    elif activate_name == 'elu':
        CPDNN_activation = tf.nn.elu
    elif activate_name == 'srelu':
        CPDNN_activation = srelu
    elif activate_name == 'sin_srelu':
        CPDNN_activation = sin_srelu
    elif activate_name == 'sin2_srelu':
        CPDNN_activation = sin2_srelu
    elif activate_name == 'slrelu':
        CPDNN_activation = slrelu
    elif activate_name == 'selu':
        CPDNN_activation = selu
    elif activate_name == 'phi':
        CPDNN_activation = phi
    # 计算 m=0 时的拟合
    Real = multilayer_DNN(input_x, Weights0, Biases0, CPDNN_activation)
    # # 计算 m=1,2,3.... 时的拟合,其中{1,2,3....}是频率数目
    for k in range(len(freqs)):
        temp_multilayer1 = multilayer_DNN(freqs[k] * input_x, Weights_COS[k], Biases_COS[k], CPDNN_activation) * tf.cos(freqs[k] * input_x)
        temp_multilayer1 = 2 * temp_multilayer1
        temp2 = multilayer_DNN(input_x, Weights_SIN[k], Biases_SIN[k], CPDNN_activation) * tf.sin(freqs[k]*input_x)
        temp2 = 2 * temp2
        Real = Real + multilayer_DNN(input_x, Weights_COS[k], Biases_COS[k], CPDNN_activation) * tf.cos(freqs[k]*input_x)\
               + multilayer_DNN(input_x, Weights_SIN[k], Biases_SIN[k], CPDNN_activation) * tf.sin(freqs[k]*input_x)
    return Real


def CPS_DNN_scale(input_x, freqs, Weights0, Biases0, Weights_COS, Biases_COS, Weights_SIN, Biases_SIN,
                  activate_name=tf.nn.relu):
    if activate_name == 'relu':
        CPDNN_activation = tf.nn.relu
    elif activate_name == 'leaky_relu':
        CPDNN_activation = tf.nn.leaky_relu(0.2)
    elif activate_name == 'elu':
        CPDNN_activation = tf.nn.elu
    elif activate_name == 'srelu':
        CPDNN_activation = srelu
    elif activate_name == 'sin_srelu':
        CPDNN_activation = sin_srelu
    elif activate_name == 'sin2_srelu':
        CPDNN_activation = sin2_srelu
    elif activate_name == 'slrelu':
        CPDNN_activation = slrelu
    elif activate_name == 'selu':
        CPDNN_activation = selu
    elif activate_name == 'phi':
        CPDNN_activation = phi
    # 计算 m=0 时的拟合
    Real = multilayer_DNN(input_x, Weights0, Biases0, CPDNN_activation)
    # # 计算 m=1,2,3.... 时的拟合
    for k in range(len(freqs)):
        temp_multilayer = multilayer_DNN(freqs[k]*input_x, Weights_COS[k], Biases_COS[k], CPDNN_activation) * tf.cos(freqs[k]*input_x)
        Real = Real + multilayer_DNN(freqs[k]*input_x, Weights_COS[k], Biases_COS[k], CPDNN_activation) * tf.cos(freqs[k]*input_x) \
               + multilayer_DNN(freqs[k]*input_x, Weights_SIN[k], Biases_SIN[k], CPDNN_activation) * tf.sin(freqs[k]*input_x)
    return Real


def CPS_DNN_residual(input_x, freqs, Weights0, Biases0, Weights_COS, Biases_COS, Weights_SIN, Biases_SIN,
                     activate_name=tf.nn.relu):
    if activate_name == 'relu':
        CPDNN_activation = tf.nn.relu
    elif activate_name == 'leaky_relu':
        CPDNN_activation = tf.nn.leaky_relu(0.2)
    elif activate_name == 'elu':
        CPDNN_activation = tf.nn.elu
    elif activate_name == 'srelu':
        CPDNN_activation = srelu
    elif activate_name == 'sin_srelu':
        CPDNN_activation = sin_srelu
    elif activate_name == 'sin2_srelu':
        CPDNN_activation = sin2_srelu
    elif activate_name == 'slrelu':
        CPDNN_activation = slrelu
    elif activate_name == 'selu':
        CPDNN_activation = selu
    elif activate_name == 'phi':
        CPDNN_activation = phi
    # 计算 m=0 时的拟合
    Real = multilayer_DNN_residual(input_x, Weights0, Biases0, CPDNN_activation)
    # # 计算 m=1,2,3.... 时的拟合
    for k in range(len(freqs)):

        Real = Real + multilayer_DNN_residual(input_x, Weights_COS[k], Biases_COS[k], CPDNN_activation) * tf.cos(freqs[k]*input_x) \
               + multilayer_DNN_residual(input_x, Weights_SIN[k], Biases_SIN[k], CPDNN_activation) * tf.sin(freqs[k]*input_x)
    return Real


# L1正则化参数
def regular_weights_L1(weights0, weights_cos, weights_sin):
    layers1 = len(weights0)
    freq_num2sin_cos = len(weights_cos)
    layers2sin_cos = len(weights_cos[0])
    regular_w = 0
    for i_layer1 in range(layers1):
        regular_w = regular_w + tf.reduce_mean(tf.abs(weights0[i_layer1]), keep_dims=False)

    for i_freq in range(freq_num2sin_cos):
        for i_layer_sin_cos in range(layers2sin_cos):
            regular_w = regular_w + tf.reduce_mean(tf.abs(weights_cos[i_freq][i_layer_sin_cos]), keep_dims=False) + \
                          tf.reduce_mean(tf.abs(weights_sin[i_freq][i_layer_sin_cos]), keep_dims=False)

    return regular_w


# L2正则化参数
def regular_weights_L2(weights0, weights_cos, weights_sin):
    layers1 = len(weights0)
    freq_num2sin_cos = len(weights_cos)
    layers2sin_cos = len(weights_cos[0])
    regular_w = 0
    for i_layer1 in range(layers1):
        regular_w = regular_w + tf.norm(weights0[i_layer1])

    for i_freq in range(freq_num2sin_cos):
        for i_layer_sin_cos in range(layers2sin_cos):
            regular_w = regular_w + tf.norm(weights_cos[i_freq][i_layer_sin_cos]) + \
                          tf.norm(weights_sin[i_freq][i_layer_sin_cos])

    return regular_w
