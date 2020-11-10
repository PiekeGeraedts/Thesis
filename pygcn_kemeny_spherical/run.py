import sys
import os
gamma_lst = [0., 0.1, 0.01, 0.001, 0.0001, 0.00001]

for gamma in gamma_lst:
	os.system(f"python3 train_SPSA1.py --epochs 500 --gamma {gamma}")