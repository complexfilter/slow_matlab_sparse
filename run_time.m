close all;clear;clc;
image = double(imread('lena.bmp'));
r = 1;
eps = 1e-16;
G_L=Laplacian_Generation_Fast(image,r,eps); 

