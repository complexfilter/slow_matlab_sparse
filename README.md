# Matlab's sparse function is slow
The function demonstrate that sparse function in Matlab is much slower than than scipy.sparse.coo_matrix.

The code is for generating a matting Laplacian matrix in paper: A Closed-Form Solution to Natural Image Matting.

1. Reproducing the runtime in Matlab:
Run run_tume.m

Result in my Desktop: Runtime of generating a sparse matrix in Matlab:1.7933 second.

2. Reprodcuing the runtime in Python:
python3 Laplacian_Generate.py

Result in my Desktop: Runtime of generating a sparse matrix via SicPy: 0.1416337490081787 second.

