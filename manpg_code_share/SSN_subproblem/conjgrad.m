function [x, k,res] = conjgrad(A,b,tol)
% CONJGRAD  Conjugate Gradient Method.
% solving A*x = b;  A is linear operator, b is n*n matrix
% A(x) is column wise mapping A: x(:,i) -> b(:,i) with A = bldiag{A1,A2,..An}
k = 0;
if nargin<3
    tol=1e-7;
end
tol = max(1e-7, tol);

x = 0;
r = b ;%- A(x);
r_norm = norm(r,'fro')^2;
if r_norm < tol^2
    res = sqrt(r_norm);
    return
end
p = r;

Ap = A(p);
%s = dot(y,z);
s = sum(sum(p.*Ap));
%t = dot(r,y)./s;
alpha = r_norm/s;
x = x + alpha*p;
r_norm_prev = r_norm;

for k = 1:numel(b)
    r = r - alpha*Ap;
    r_norm = norm(r,'fro')^2;
    if r_norm <tol^2
        res = sqrt(r_norm);
        return;
    end
    %B = dot(r,z)./s;
    Beta = r_norm/r_norm_prev;
    p = r + Beta*p;
    Ap = A(p);
    s = sum(sum(p.*Ap));
    alpha = r_norm/s;
    x = x + alpha*p;
    r_norm_prev = r_norm;
end
end
