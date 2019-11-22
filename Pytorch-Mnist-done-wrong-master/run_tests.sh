#!/bin/bash
cd competition
mkdir -p results/final results/test results/val
cd ..
python3 Practica_3.py --target_class 0 --epochs 9
python3 Practica_3.py --target_class 1 --epochs 9
python3 Practica_3.py --target_class 2 --epochs 9
python3 Practica_3.py --target_class 3 --epochs 9
python3 Practica_3.py --target_class 4 --epochs 9
python3 Practica_3.py --target_class 5 --epochs 9
python3 Practica_3.py --target_class 6 --epochs 9
python3 Practica_3.py --target_class 7 --epochs 9
python3 Practica_3.py --target_class 8 --epochs 9
python3 Practica_3.py --target_class 9 --epochs 9 --concat











