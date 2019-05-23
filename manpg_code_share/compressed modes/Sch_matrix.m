function [H] = Sch_matrix(a,b,l1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Schrodinger operator
% Input: 
% a, b: Two end points. %
% n: number of grid points. %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% h = (b-a)/(n); h1=h*h;
% H = zeros(n,n);
% for i=1:n-1
% H(i,i) = -2/h1; H(i+1,i) = 1/h1; H(i,i+1)= 1/h1;
% end
% H(n,n) = -2/h1;
% H(1,n)= 1/h1;  H(n,1)=1/h1;

dx = (b-a)/l1;
Lap_1D = -2*speye(l1,l1) + spdiags(ones(l1-1,1),-1,l1,l1) + spdiags(ones(l1,1),1,l1,l1);
Lap_1D(1,l1) = 1;  Lap_1D(l1,1) = 1;
H = -1/2*Lap_1D/dx/dx ;

