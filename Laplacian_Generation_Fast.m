% The code is from alpahmatting.com website
function L=Laplacian_Generation_Fast(I,WinSizeHalf,epsilon)
%% Input
%  I: Height*Width the Laplacian of a PAN image. The pixel values of I
%  is within the range of [0,1].
%% Output 
%  L: (Height*Width)*(Height*Width) sparse graph laplacian matrix
I=double(I);
[Height,Width]=size(I);
NumPixels=Height*Width;
WinCardinality=(2*WinSizeHalf+1)^2; 
% WinCardinality refers to |W_k| in paper, i.e. the cardinality of the
% window.
Ind_Array_Pixel=reshape(1:NumPixels,Height,Width);
IndPixel = im2col(Ind_Array_Pixel,[2*WinSizeHalf+1 2*WinSizeHalf+1],'sliding');
wins_number = (Height-2*WinSizeHalf)*(Width-2*WinSizeHalf);
winI = I(IndPixel);
win_mu = mean(winI,1);
win_var = 1/WinCardinality*sum(winI.*winI,1)-win_mu.*win_mu;
inv_item = 1./(win_var + (epsilon/WinCardinality));
X = (winI - repmat(win_mu,WinCardinality,1)).*repmat(inv_item,WinCardinality,1);
F = reshape(X,WinCardinality,1,wins_number).*reshape((winI - repmat(win_mu,WinCardinality,1)),1,WinCardinality,wins_number);
Vals = eye(WinCardinality)-1/WinCardinality*(1+F);
RowInd = repmat(IndPixel,[WinCardinality,1]);
ColInd = repelem(IndPixel,WinCardinality,1);
Vals=Vals(:);
tStart = tic;          
L=sparse(RowInd(:),ColInd(:),Vals(:),NumPixels,NumPixels,wins_number*WinCardinality^2);
tEnd = toc(tStart);
disp(['Runtime of generating a sparse matrix in Matlab:', num2str(tEnd), ' second.']);
end