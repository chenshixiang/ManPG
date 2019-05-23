function [ x_prox, delta ,Inact_set] = prox_l21(  b ,lambda,r )
%proximal mapping of L_21 norm
[n,r] = size(b);  delta = cell(n,1);
nr = vecnorm(b,2,2) ;
a =  nr - lambda;
if r < 15
    Act_set = double( a > 0);
else
    Act_set = ( a > 0);
end

x_prox = Act_set.*(1- lambda./nr).*b;
for i = 1:n
    delta{i} = (eye(r) - lambda./nr(i).*(eye(r) - b(i,:)'*b(i,:)./nr(i)))*Act_set(i);
end
% diag = 1 - lambda*ones(size(b))./nr + lambda.*b.^2./(nr.^3);
% diag = Act_set.*diag;
if nargout==3
    Inact_set= (a <= 0);
end

end

