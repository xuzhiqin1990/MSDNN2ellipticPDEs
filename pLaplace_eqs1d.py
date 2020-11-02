import tensorflow as tf
import numpy as np


def get_infos_2laplace(in_dim=None, out_dim=None, region_a=0, region_b=1, p=2, eps=0.01):
    f = lambda x: tf.ones_like(x)

    aeps = lambda x: 1.0 / (2 + tf.cos(2 * np.pi * x / eps))

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    utrue = lambda x: x - tf.square(x) + eps * (
                1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
            np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    return utrue, f, aeps, u_l, u_r


def get_infos_3laplace(in_dim=None, out_dim=None, region_a=0, region_b=1, p=2, eps=0.01):
    f = lambda x: abs(2 * x - 1) * (
                4 * eps + 2 * eps * tf.cos(2 * np.pi * x / eps) + np.pi * (1 - 2 * x) * tf.sin(2 * np.pi * x / eps)) / (
                              2 * eps)

    aeps = lambda x: 1.0 / (2 + tf.cos(2 * np.pi * x / eps))

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    utrue = lambda x: x - tf.square(x) + eps * (
                1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
            np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    return utrue, f, aeps, u_l, u_r


def get_infos_4laplace(in_dim=None, out_dim=None, region_a=0, region_b=1, p=2, eps=0.01):
    f = lambda x: ((1-2*x)**2) * (2+tf.cos(2*np.pi*x/eps))*(
                6 * eps + 3 * eps * tf.cos(2 * np.pi * x / eps) - 2*np.pi * (2 * x-1) * tf.sin(2 * np.pi * x / eps)) / (
                              4 * eps)
    aeps = lambda x: 1.0 / (2 + tf.cos(2 * np.pi * x / eps))

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    utrue = lambda x: x - tf.square(x) + eps * (
                1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
            np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    return utrue, f, aeps, u_l, u_r


def get_infos_5laplace(in_dim=None, out_dim=None, region_a=0, region_b=1, p=2, eps=0.01):
    # f = lambda x: ((2 * x - 1) ** 3) * ((2 + tf.cos(2 * np.pi * x / eps)) ** 2) * (
    #         3 * np.pi * (2 * x - 1) * tf.sin(2 * np.pi * x / eps) - 4 * eps * tf.cos(2 * np.pi * x / eps) - 8 * eps) / (
    #                       8 * eps)
    # f = lambda x: ((1-2 * x ) ** 3) * ((2 + tf.cos(2 * np.pi * x / eps)) ** 2) * (
    #         3 * np.pi * (2 * x - 1) * tf.sin(2 * np.pi * x / eps) - 4 * eps * tf.cos(2 * np.pi * x / eps) - 8 * eps) / (
    #                       8 * eps)
    f = lambda x: -1.0*abs((2 * x - 1) ** 3) * ((2 + tf.cos(2 * np.pi * x / eps))**2) * (
            3 * np.pi * (2 * x - 1) * tf.sin(2 * np.pi * x / eps) - 4 * eps * tf.cos(2 * np.pi * x / eps) - 8*eps) / (
                          8 * eps)

    aeps = lambda x: 1.0 / (2 + tf.cos(2 * np.pi * x / eps))

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    utrue = lambda x: x - tf.square(x) + eps * (
                1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
            np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    return utrue, f, aeps, u_l, u_r


def get_infos_8laplace(in_dim=None, out_dim=None, region_a=0, region_b=1, p=2, eps=0.01):
    f = lambda x: ((1 - 2 * x) ** 6) * ((2 + tf.cos(2 * np.pi * x / eps)) ** 5) * (
            7 * eps * tf.cos(2 * np.pi * x / eps) + 2 * (
                7 * eps - 3 * np.pi * (2 * x - 1) * tf.sin(2 * np.pi * x / eps))) / (
                          64 * eps)

    aeps = lambda x: 1.0 / (2 + tf.cos(2 * np.pi * x / eps))

    u_l = lambda x: tf.zeros_like(x)
    u_r = lambda x: tf.zeros_like(x)

    utrue = lambda x: x - tf.square(x) + eps * (
                1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
            np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    return utrue, f, aeps, u_l, u_r


def get_infos_multi_scale(in_dim=None, out_dim=None, region_a=0, region_b=1, p=2, eps=0.01):
    f = lambda x: (np.power(1 - 2 * x, p) * np.power(2 + tf.cos(2 * np.pi * x / eps), p)*(
                eps * (p - 1) * (2+tf.cos(2 * np.pi * x / eps)) - np.pi * (p - 2) * (2 * x - 1) * tf.sin(2 * np.pi * x / eps))) / (
                              np.power(2, p - 2) * eps * ((1 - 2 * x) ** 2) * ((2 + tf.cos(2 * np.pi * x / eps)) ** 3))

    aeps = lambda x: 1.0 / (2 + tf.cos(2 * np.pi * x / eps))

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    utrue = lambda x: x - tf.square(x) + eps * (
            1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
        np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    return utrue, f, aeps, u_l, u_r


def get_infos__multi_scale_abs(in_dim=None, out_dim=None, region_a=0, region_b=1, p=2, eps=0.01):

    f = lambda x: (np.power(abs(1 - 2 * x), p) * np.power(2 + tf.cos(2 * np.pi * x / eps), p) * (
            eps * (p - 1) * (2 + tf.cos(2 * np.pi * x / eps)) - np.pi * (p - 2) * (2 * x - 1) * tf.sin(
        2 * np.pi * x / eps))) / (
                          np.power(2, p - 2) * eps * ((1 - 2 * x) ** 2) * ((2 + tf.cos(2 * np.pi * x / eps)) ** 3))

    aeps = lambda x: 1 / (2 + tf.cos(2 * np.pi * x / eps))

    u_l = lambda x: tf.zeros_like(x)

    u_r = lambda x: tf.zeros_like(x)

    u_true = lambda x: x - tf.square(x) + eps * (
            1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
        np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    return u_true, f, aeps, u_l, u_r


def get_infos_pLaplace(in_dim=None, out_dim=None, region_a=0, region_b=1, p=2, eps=0.01, eqs_name=None):
    if eqs_name == 'multi_scale':
        f = lambda x: (np.power(abs(1 - 2 * x), p) * np.power(2 + tf.cos(2 * np.pi * x / eps), p) * (
                eps * (p - 1) * (2 + tf.cos(2 * np.pi * x / eps)) - np.pi * (p - 2) * (2 * x - 1) * tf.sin(
            2 * np.pi * x / eps))) / (
                              np.power(2, p - 2) * eps * ((1 - 2 * x) ** 2) * ((2 + tf.cos(2 * np.pi * x / eps)) ** 3))

        aeps = lambda x: 1 / (2 + tf.cos(2 * np.pi * x / eps))

        u_l = lambda x: tf.zeros_like(x)

        u_r = lambda x: tf.zeros_like(x)

        utrue = lambda x: x - tf.square(x) + eps * (
                1 / np.pi * tf.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * tf.cos(
            np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

        return utrue, f, aeps, u_l, u_r


