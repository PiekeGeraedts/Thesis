"""
Purpose:
	Small script to call train multiple times. Handy for hyperparameterization.
Date:
	15-11-2020
"""
import sys
import os

'''
os.system(f"python3 KemenyOptimisation.py --subtract --epochs 1000 --lr 0.01 --clipper")
os.system(f"python3 KemenyOptimisation.py --subtract --epochs 1000 --lr 0.01")
os.system(f"python3 KemenyOptimisation.py --subtract --epochs 1000 --lr 0.05 --clipper")
os.system(f"python3 KemenyOptimisation.py --subtract --epochs 1000 --lr 0.05")
os.system(f"python3 KemenyOptimisation.py --subtract --epochs 1000 --lr 0.005 --clipper")
os.system(f"python3 KemenyOptimisation.py --subtract --epochs 1000 --lr 0.005")

os.system(f"python3 KemenyOptimisation.py --squared --epochs 1000 --lr 0.01 --clipper")
os.system(f"python3 KemenyOptimisation.py --squared --epochs 1000 --lr 0.01")
os.system(f"python3 KemenyOptimisation.py --squared --epochs 1000 --lr 0.05 --clipper")
os.system(f"python3 KemenyOptimisation.py --squared --epochs 1000 --lr 0.05")
os.system(f"python3 KemenyOptimisation.py --squared --epochs 1000 --lr 0.005 --clipper")
os.system(f"python3 KemenyOptimisation.py --squared --epochs 1000 --lr 0.005")

os.system(f"python3 KemenyOptimisation.py --softmax --epochs 1000 --lr 0.01 --clipper")
os.system(f"python3 KemenyOptimisation.py --softmax --epochs 1000 --lr 0.01")
os.system(f"python3 KemenyOptimisation.py --softmax --epochs 1000 --lr 0.05 --clipper")
os.system(f"python3 KemenyOptimisation.py --softmax --epochs 1000 --lr 0.05")
os.system(f"python3 KemenyOptimisation.py --softmax --epochs 1000 --lr 0.005 --clipper")
os.system(f"python3 KemenyOptimisation.py --softmax --epochs 1000 --lr 0.005")

os.system(f"python3 KemenyOptimisationSpherical.py --epochs 1000 --lr 0.01 --clipper")
os.system(f"python3 KemenyOptimisationSpherical.py --epochs 1000 --lr 0.01")
os.system(f"python3 KemenyOptimisationSpherical.py --epochs 1000 --lr 0.05 --clipper")
os.system(f"python3 KemenyOptimisationSpherical.py --epochs 1000 --lr 0.05")
os.system(f"python3 KemenyOptimisationSpherical.py --epochs 1000 --lr 0.005 --clipper")
os.system(f"python3 KemenyOptimisationSpherical.py --epochs 1000 --lr 0.005")
'''


os.system(f"python3 train_SPSA.py --subtract --epochs 200 --nlayer 6 --lrV 0.01 --fastmode")
os.system(f"python3 train_SPSA.py --subtract--epochs 200 --nlayer 6 --lrV 0.05 --fastmode")
os.system(f"python3 train_SPSA.py --subtract --epochs 200 --nlayer 6 --lrV 0.1 --fastmode")
os.system(f"python3 train_SPSA.py --subtract --epochs 200 --nlayer 6 --lrV 0.15 --fastmode")
os.system(f"python3 train_SPSA.py --subtract --epochs 200 --nlayer 6 --lrV 0.3 --fastmode")

#os.system(f"python3 train_SPSA.py --softmax --epochs 100 --N 200 --nlayer 6")
#os.system(f"python3 train_SPSA.py --subtract --epochs 100 --N 200 --nlayer 6")
#os.system(f"python3 train_SPSA.py --spherical --epochs 100 --N 200 --nlayer 6")




