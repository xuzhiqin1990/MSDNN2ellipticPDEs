import tensorflow as tf
import numpy as np


#  例一 u=e^(-x)*(x+y^3)
def true_solution_PDE1(input_dim=None, output_dim=None):
    u_true = lambda x, y: (tf.exp(-1.0*x))*(x + tf.pow(y, 3))
    return u_true


def force_side2PDE1(input_dim=None, output_dim=None):
    f_side = lambda x, y: -(tf.exp(-1.0*x)) * (x - 2 + tf.pow(y, 3) + 6 * y)
    return f_side


def boundary2PDE1(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    ux_left = lambda x, y: tf.exp(-left_bottom)*(tf.pow(y, 3) + 1.0*left_bottom)
    ux_right = lambda x, y: tf.exp(-right_top)*(tf.pow(y, 3) + 1.0*right_top)
    uy_bottom = lambda x, y: tf.exp(-x)*(tf.pow(left_bottom, 3) + x)
    uy_top = lambda x, y: tf.exp(-x)*(tf.pow(right_top, 3) + x)
    return ux_left, ux_right, uy_bottom, uy_top


# 例二 u=y^2*sin(pi*x)
def true_solution_PDE2(input_dim=None, output_dim=None):
    u_true = lambda x, y: tf.square(y)*tf.sin(np.pi*x)
    return u_true


def force_side2PDE2(input_dim=None, output_dim=None):
    f_side = lambda x, y: (-1.0)*tf.sin(np.pi*x) * (2 - np.square(np.pi)*tf.square(y))
    return f_side


def boundary2PDE2(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    ux_left = lambda x, y: tf.square(y)*tf.sin(np.pi*left_bottom)
    ux_right = lambda x, y: tf.square(y)*tf.sin(np.pi*right_top)
    uy_bottom = lambda x, y: tf.square(left_bottom)*tf.sin(np.pi*x)
    uy_top = lambda x, y: tf.square(right_top)*tf.sin(np.pi*x)
    return ux_left, ux_right, uy_bottom, uy_top


# 例三 u=(e^x)*e^(y)
def true_solution_PDE3(input_dim=None, output_dim=None):
    u_true = lambda x, y: tf.exp(x)*tf.exp(y)
    return u_true


def force_side2PDE3(input_dim=None, output_dim=None):
    f_side = lambda x, y: -2.0*(tf.exp(x)*tf.exp(y))
    return f_side


def boundary2PDE3(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    ux_left = lambda x, y: tf.multiply(tf.exp(y), tf.exp(left_bottom))
    ux_right = lambda x, y: tf.multiply(tf.exp(y), tf.exp(right_top))
    uy_bottom = lambda x, y: tf.multiply(tf.exp(x), tf.exp(left_bottom))
    uy_top = lambda x, y: tf.multiply(tf.exp(x), tf.exp(right_top))
    return ux_left, ux_right, uy_bottom, uy_top


# 例四 u=1/2*(x^2+y^2)
def true_solution_PDE4(input_dim=None, output_dim=None):
    u_true = lambda x, y: 0.25*(tf.pow(x, 2)+tf.pow(y, 2))
    return u_true


def force_side2PDE4(input_dim=None, output_dim=None):
    f_side = lambda x, y: -1.0*tf.ones_like(x)
    return f_side


def boundary2PDE4(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    ux_left = lambda x, y: 0.25*tf.pow(y, 2) + 0.25*tf.pow(left_bottom, 2)
    ux_right = lambda x, y: 0.25*tf.pow(y, 2) + 0.25*tf.pow(right_top, 2)
    uy_bottom = lambda x, y: 0.25*tf.pow(x, 2) + 0.25*tf.pow(left_bottom, 2)
    uy_top = lambda x, y: 0.25*tf.pow(x, 2) + 0.25*tf.pow(right_top, 2)
    return ux_left, ux_right, uy_bottom, uy_top


# 例五  u=1/2*(x^2+y^2)+x+y
def true_solution_PDE5(input_dim=None, output_dim=None):
    u_true = lambda x, y: 0.25*(tf.pow(x, 2)+tf.pow(y, 2)) + x + y
    return u_true


def force_side2PDE5(input_dim=None, output_dim=None):
    f_side = lambda x, y: -1.0*tf.ones_like(x)
    return f_side


def boundary2PDE5(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    ux_left = lambda x, y: 0.25*tf.pow(y, 2) + 0.25*tf.pow(left_bottom, 2) + left_bottom + y
    ux_right = lambda x, y: 0.25*tf.pow(y, 2) + 0.25*tf.pow(right_top, 2) + right_top + y
    uy_bottom = lambda x, y: 0.25*tf.pow(x, 2) + tf.pow(left_bottom, 2) + left_bottom + x
    uy_top = lambda x, y: 0.25*tf.pow(x, 2) + 0.25*tf.pow(right_top, 2) + right_top + x
    return ux_left, ux_right, uy_bottom, uy_top


# 例六 u=(1/2)*(x^2)*(y^2)
def true_solution_PDE6(input_dim=None, output_dim=None):
    u_true = lambda x, y: 0.5 * (tf.pow(x, 2) * tf.pow(y, 2))
    return u_true


def force_side2PDE6(input_dim=None, output_dim=None):
    f_side = lambda x, y: -1.0*(tf.pow(x, 2)+tf.pow(y, 2))
    return f_side


def boundary2PDE6(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    ux_left = lambda x, y: 0.5*(tf.pow(left_bottom, 2)*tf.pow(y, 2))
    ux_right = lambda x, y: 0.5*(tf.pow(right_top, 2)*tf.pow(y, 2))
    uy_bottom = lambda x, y: 0.5*(tf.pow(x, 2)*tf.pow(left_bottom, 2))
    uy_top = lambda x, y: 0.5*(tf.pow(x, 2)*tf.pow(right_top, 2))
    return ux_left, ux_right, uy_bottom, uy_top


# 例七 u=(1/2)*(x^2)*(y^2)+x+y
def true_solution_PDE7(input_dim=None, output_dim=None):
    u_true = lambda x, y: 0.5*(tf.pow(x, 2)*tf.pow(y, 2)) + x*tf.ones_like(x) + y*tf.ones_like(y)
    return u_true


def force_side2PDE7(input_dim=None, output_dim=None):
    f_side = lambda x, y: -1.0*(tf.pow(x, 2)+tf.pow(y, 2))
    return f_side


def boundary2PDE7(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    ux_left = lambda x, y: 0.5*tf.multiply(tf.pow(left_bottom, 2), tf.pow(y, 2)) + left_bottom + y
    ux_right = lambda x, y: 0.5*tf.multiply(tf.pow(right_top, 2), tf.pow(y, 2)) + right_top + y
    uy_bottom = lambda x, y: 0.5*tf.multiply(tf.pow(x, 2), tf.pow(left_bottom, 2)) + x + left_bottom
    uy_top = lambda x, y: 0.5*tf.multiply(tf.pow(x, 2), tf.pow(right_top, 2)) + x + right_top
    return ux_left, ux_right, uy_bottom, uy_top


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数
def get_general_laplace_infos(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, laplace_name=None):
    if laplace_name == 'PDE1':
        # u=exp(-x)(x_y^3), f = -exp(-x)(x-2+y^3+6y)
        f_side = lambda x, y: -(tf.exp(-1.0*x)) * (x - 2 + tf.pow(y, 3) + 6 * y)

        u_true = lambda x, y: (tf.exp(-1.0*x))*(x + tf.pow(y, 3))

        ux_left = lambda x, y: tf.exp(-left_bottom) * (tf.pow(y, 3) + 1.0 * left_bottom)
        ux_right = lambda x, y: tf.exp(-right_top) * (tf.pow(y, 3) + 1.0 * right_top)
        uy_bottom = lambda x, y: tf.exp(-x) * (tf.pow(left_bottom, 3) + x)
        uy_top = lambda x, y: tf.exp(-x) * (tf.pow(right_top, 3) + x)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif laplace_name == 'PDE2':
        f_side = lambda x, y: (-1.0)*tf.sin(np.pi*x) * (2 - np.square(np.pi)*tf.square(y))

        u_true = lambda x, y: tf.square(y)*tf.sin(np.pi*x)

        ux_left = lambda x, y: tf.square(y) * tf.sin(np.pi * left_bottom)
        ux_right = lambda x, y: tf.square(y) * tf.sin(np.pi * right_top)
        uy_bottom = lambda x, y: tf.square(left_bottom) * tf.sin(np.pi * x)
        uy_top = lambda x, y: tf.square(right_top) * tf.sin(np.pi * x)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif laplace_name == 'PDE3':
        # u=exp(x+y), f = -2*exp(x+y)
        f_side = lambda x, y: -2.0*(tf.exp(x)*tf.exp(y))
        u_true = lambda x, y: tf.exp(x)*tf.exp(y)
        ux_left = lambda x, y: tf.multiply(tf.exp(y), tf.exp(left_bottom))
        ux_right = lambda x, y: tf.multiply(tf.exp(y), tf.exp(right_top))
        uy_bottom = lambda x, y: tf.multiply(tf.exp(x), tf.exp(left_bottom))
        uy_top = lambda x, y: tf.multiply(tf.exp(x), tf.exp(right_top))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif laplace_name == 'PDE4':
        # u=(1/4)*(x^2+y^2), f = -1
        f_side = lambda x, y: -1.0*tf.ones_like(x)
        u_true = lambda x, y: 0.25*(tf.pow(x, 2)+tf.pow(y, 2))
        ux_left = lambda x, y: 0.25 * tf.pow(y, 2) + 0.25 * tf.pow(left_bottom, 2)
        ux_right = lambda x, y: 0.25 * tf.pow(y, 2) + 0.25 * tf.pow(right_top, 2)
        uy_bottom = lambda x, y: 0.25 * tf.pow(x, 2) + 0.25 * tf.pow(left_bottom, 2)
        uy_top = lambda x, y: 0.25 * tf.pow(x, 2) + 0.25 * tf.pow(right_top, 2)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif laplace_name == 'PDE5':
        # u=(1/4)*(x^2+y^2)+x+y, f = -1
        f_side = lambda x, y: -1.0*tf.ones_like(x)

        u_true = lambda x, y: 0.25*(tf.pow(x, 2)+tf.pow(y, 2)) + x + y

        ux_left = lambda x, y: 0.25 * tf.pow(y, 2) + 0.25 * tf.pow(left_bottom, 2) + left_bottom + y
        ux_right = lambda x, y: 0.25 * tf.pow(y, 2) + 0.25 * tf.pow(right_top, 2) + right_top + y
        uy_bottom = lambda x, y: 0.25 * tf.pow(x, 2) + tf.pow(left_bottom, 2) + left_bottom + x
        uy_top = lambda x, y: 0.25 * tf.pow(x, 2) + 0.25 * tf.pow(right_top, 2) + right_top + x
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif laplace_name == 'PDE6':
        # u=(1/2)*(x^2)*(y^2), f = -(x^2+y^2)
        f_side = lambda x, y: -1.0*(tf.pow(x, 2)+tf.pow(y, 2))

        u_true = lambda x, y: 0.5 * (tf.pow(x, 2) * tf.pow(y, 2))

        ux_left = lambda x, y: 0.5 * (tf.pow(left_bottom, 2) * tf.pow(y, 2))
        ux_right = lambda x, y: 0.5 * (tf.pow(right_top, 2) * tf.pow(y, 2))
        uy_bottom = lambda x, y: 0.5 * (tf.pow(x, 2) * tf.pow(left_bottom, 2))
        uy_top = lambda x, y: 0.5 * (tf.pow(x, 2) * tf.pow(right_top, 2))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif laplace_name == 'PDE7':
        # u=(1/2)*(x^2)*(y^2)+x+y, f = -(x^2+y^2)
        f_side = lambda x, y: -1.0*(tf.pow(x, 2)+tf.pow(y, 2))

        u_true = lambda x, y: 0.5*(tf.pow(x, 2)*tf.pow(y, 2)) + x*tf.ones_like(x) + y*tf.ones_like(y)

        ux_left = lambda x, y: 0.5 * tf.multiply(tf.pow(left_bottom, 2), tf.pow(y, 2)) + left_bottom + y
        ux_right = lambda x, y: 0.5 * tf.multiply(tf.pow(right_top, 2), tf.pow(y, 2)) + right_top + y
        uy_bottom = lambda x, y: 0.5 * tf.multiply(tf.pow(x, 2), tf.pow(left_bottom, 2)) + x + left_bottom
        uy_top = lambda x, y: 0.5 * tf.multiply(tf.pow(x, 2), tf.pow(right_top, 2)) + x + right_top
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top




