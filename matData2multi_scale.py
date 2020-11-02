# !python3
# -*- coding: utf-8 -*-
# author: flag

import numpy as np
import scipy.io


# load the data from matlab of .mat
def loadMatlabIdata(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def get_data2multi_scale(equation_name=None, mesh_number=2):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = 'data2Matlab/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = 'data2Matlab/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = 'data2Matlab/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = 'data2Matlab/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = 'data2Matlab/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_XY = loadMatlabIdata(test_meshXY_file)
    XY = mesh_XY['meshXY']
    test_xy_data = np.transpose(XY, (1, 0))
    return test_xy_data


if __name__ == '__main__':
    mat_file_name = 'data2Matlab/meshXY.mat'
    mat_data = loadMatlabIdata(mat_file_name)
    XY = mat_data['meshXY']
    XY_T = np.transpose(XY, (1, 0))
    print('shdshd')