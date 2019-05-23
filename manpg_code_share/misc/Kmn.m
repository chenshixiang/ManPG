function K = Kmn(m,n)

 K = zeros(m*n, m*n);
 m0 = 1:(m*n);

N = reshape(m0, m,n)';
n0 = N(:);

for i = 1:(m*n)
K(m0(i), n0(i)) = 1;
end 