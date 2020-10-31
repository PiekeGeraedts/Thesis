import sys
import os
gamma_lst = [1000, 100, 10, 1 ,0., 0.1, 0.01, 0.001, 0.0001, 0.00001]

for gamma in gamma_lst:
	os.system(f"python3 train_SPSA.py --epochs 300 --gamma {gamma}")
