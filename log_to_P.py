'''
Author: 
    Pieke
Purpose:
    Write transition matrix in file. Problem: to many values. Can use sparcity to only keep non-zero values and their location.
    A script that converts a log file (check adjacency.log or P.log for format) into a .npy file 
Date created:
    25-06-2020
'''

import pygcn
import numpy as np
import sys

argv = sys.argv
if len(argv) == 3:
    log_file = open(argv[1], "r")
    n = int (argv[2])
    P = np.zeros((n,n))
    row_lst = []
    col_lst = []
    for line in log_file:
        row, data = line.split('-')
        for str in data.split(' '):
            try:
                col, val = str.split(':')
                P[int(row), int(col)] = float(val)
            except:
                #probably an empty line
                pass
    np.save('TransitionMatrix.npy', P)
    #print (P)
else:
    print ('REQUIRED INPUT: file_path dimension')
        

