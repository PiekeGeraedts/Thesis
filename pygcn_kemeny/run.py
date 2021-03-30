"""
Purpose:
	Small script to call train multiple times. Handy for hyperparameterization.
Date:
	15-11-2020
"""
import sys
import os

# softmax 
os.system(f"python3 train_SPSA.py --softmax --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 1")
os.system(f"python3 train_SPSA.py --softmax --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 12")
os.system(f"python3 train_SPSA.py --softmax --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 123")
# squared
os.system(f"python3 train_SPSA.py --squared --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 1")
os.system(f"python3 train_SPSA.py --squared --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 12")
os.system(f"python3 train_SPSA.py --squared --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 123")
# subtract
os.system(f"python3 train_SPSA.py --subtract --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 1")
os.system(f"python3 train_SPSA.py --subtract --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 12")
os.system(f"python3 train_SPSA.py --subtract --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 123")
# spherical
os.system(f"python3 train_SPSA.py --spherical --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 1")
os.system(f"python3 train_SPSA.py --spherical --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 12")
os.system(f"python3 train_SPSA.py --spherical --epochs 200 --N 200 --clipper --fastmode --nlayer 6 --seed 123")




#os.system(f"python3 KemenyOptimisation.py --softmax --epochs 300 --eps 0.01")
#os.system(f"python3 KemenyOptimisation.py --squared --epochs 300 --eps 0.01")
#os.system(f"python3 KemenyOptimisation.py --subtract --epochs 300 --eps 0.01")
#os.system(f"python3 KemenyOptimisationSpherical.py --epochs 300 --eps 0.01")
