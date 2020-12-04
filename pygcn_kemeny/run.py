import sys
import os
gamma_lst = [1000, 100, 10, 1 ,0., 0.1, 0.01, 0.001, 0.0001, 0.00001]

#for gamma in gamma_lst:
#	os.system(f"python3 train_SPSA.py --epochs 500 --gamma {gamma}")


os.system(f"python3 train_SPSA.py --epochs 500")
os.system(f"python3 train_SPSA.py --epochs 500 --lr 0.001")
os.system(f"python3 train_SPSA.py --epochs 500 --clipper")
os.system(f"python3 train_SPSA.py --epochs 500 --eps 0.0")

os.system(f"python3 train_SPSA.py --epochs 1000")
os.system(f"python3 train_SPSA.py --epochs 1000 --clipper")
os.system(f"python3 train_SPSA.py --epochs 1000 --eps 0.0")
os.system(f"python3 train_SPSA.py --epochs 1000 --eps 0.0 --lr 0.001")

'''
#epochs=1000 folder
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.001")	#constant eta
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.001 --linear")
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.001 --sqrt")
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.001 --log")

os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.01")	#constant eta
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.01 --linear")
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.01 --sqrt")
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.01 --log")

os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.0")	#constant eta
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.0 --linear")
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.0 --sqrt")
os.system(f"python3 test.py --epochs 1000 --lr 0.01 --eps 0.0 --log")

#epochs=3000 folder
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.0001")	#constant eta
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.0001 --linear")
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.0001 --sqrt")
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.0001 --log")

os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.001")	#constant eta
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.001 --linear")
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.001 --sqrt")
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.001 --log")

os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.0")	#constant eta
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.0 --linear")
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.0 --sqrt")
os.system(f"python3 test.py --epochs 3000 --lr 0.001 --eps 0.0 --log")
'''