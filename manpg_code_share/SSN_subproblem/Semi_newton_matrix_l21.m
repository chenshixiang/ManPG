function [Z,inner_iter,Lam,r_l,stop_flag] = Semi_newton_matrix_l21(n,r,X,t, B,mut,inner_tol,prox,Lam0)
%find the zero of X'*Z(Lam)+ Z(Lam)'*X=2*I, where z(Lam) is the proximal
%point of B+2*t*X*Lam      %prox_number=0;
XtX = eye(r);
Xt  = X';
global  Dn pDn
numb=0; stop_flag=0;
% commat = Kmn(n,n);
%[n,p]=size(X);
if nargin==8
    Lam=zeros(r);
else
    Lam=Lam0;
end
X_Lam_prod = B+ 2*t*X*Lam;
[Z,delta,Inact_set] = prox(X_Lam_prod,mut,r);
ZX = Z'*X;
R_Lam= ZX+ZX'-2*eye(r);
RE=pDn*R_Lam(:);
%  s = s - r_s/gd;
r_l=norm(R_Lam,'fro');

%g=sparse(n^2,n^2);
lambda=0.1; eta1=0.1; eta2=0.3;  sigma=r_l; lambda0=0.001;
inner_iter=1;
while  r_l^2 > inner_tol
    g=zeros(r^2);
    reg= lambda*max(min(r_l,0.2),1e-11);
    %   nnzZ = nnz(Z);
    %if r < 15
    %        if nnzZ > floor(r*(r+1)/2) % directly compute inverse of G
    for i=1:r
        for j = 1:r
            for k = 1:n
                g((i*r-r+1:i*r),(j*r-r+1:j*r)) =g((i*r-r+1:i*r),(j*r-r+1:j*r)) + X(k,i)*X(k,j)*delta{k};
            end
        end
    end
    G = 4*t*pDn*(g*Dn);
    % new_d =-(G+ reg*eye(n*(n+1)/2))\RE;
    [LG,UG] = lu(G+ reg*eye(r*(r+1)/2));% use chol decompostion  solve linear system
    %new_d2 = -(4*t*Dn*pDn* g+ reg*eye(r^2))\R_Lam(:);
    %new_d2 = reshape(new_d2,r,r);
    new_d = -(UG\(LG\RE));
    %
    %         else% Use SWM Formula  dim: number of non zero of Z
    %             Xstack = zeros(nnzZ,r^2);
    %             dim  =0;
    %             for i = 1:r
    %                 row = find(delta(:,i)==1);
    %                 Xstack((dim+1:dim+length(row)) ,(i*r-r+1:i*r)) =  X(row,:);
    %                 dim = length(row) + dim;
    %             end
    %             V = Xstack*Dn;   U =4*t* pDn*Xstack';
    %             [LG,UG] = lu(eye(nnzZ) + 1/reg*V*U);
    %             new_d = -(1/reg*RE - 1/reg^2 *U*(UG\(LG\(V*RE))));
    %             %new_d = -(1/reg*RE - 1/reg^2 *U*((eye(nnzZ) + 1/reg*V*U)\(V*RE)));
    %         end
    new_d=Dn*new_d;
    norm_d=norm(new_d)^2;
    new_d=reshape(new_d,r,r);
    %     else% implement cg
    %         Blkd = cell(r,1);
    %         for i=1:r
    %            % Blkd{i} = X'*(Act_set(:,i).*X);
    %             ind  =delta(:,i);
    %             if sum(ind) < n/2
    %              X_ind = X(ind,:);
    %              Blkd{i} = X_ind'*X_ind;
    %             else
    %                 ind = Inact_set(:,i);
    %                % X_ind = X(ind,:);
    %                 X_ind = Xt(:,ind);
    %                 Blkd{i} = XtX -  X_ind*X_ind';
    %             end
    %         end
    %         fun =@(x) linop(Blkd,x,r,t,reg);
    %         [new_d,cg_iter,res] = conjgrad(fun,-R_Lam,min(1e-4,1e-3*r_l));
    %         norm_d=norm(new_d)^2;
    %     end
    
    X_d_prod=2*t*X*new_d;
    X_Lam_new_prod= X_Lam_prod + X_d_prod;
    
    Lam_new = Lam + new_d ;
    [Z, delta,Inact_set]=prox(X_Lam_new_prod,mut,r);
    ZX=Z'*X;
    R_Lam_new= ZX+ZX'-2*eye(r);
    numb=numb+1;
    r_l_new=norm(R_Lam_new,'fro');
    % while  r_l_new ^2 >=  r_l^2 *(1-  armjp *t_new) && t_new > 1e-3
    
    pho=-sum(sum(R_Lam_new.*new_d))/norm_d;
    if r_l_new<=0.99*sigma
        Lam = Lam_new;
        r_l=r_l_new;  sigma=r_l;  R_Lam=R_Lam_new; RE=pDn*R_Lam(:);
        X_Lam_prod = X_Lam_new_prod;
    else
        V=Lam- sum(sum(R_Lam_new.*(Lam-Lam_new)))/r_l_new^2* R_Lam_new;
        X_V_prod=B+2*t*X*V;
        [Z,delta,Inact_set]=prox(X_V_prod,mut,r);
        ZX=Z'*X;   R_V= ZX+ZX'-2*eye(r);  %RE=En*R_Lam(:);
        r_v=norm(R_V,'fro');
        numb = numb+1;
        if pho>=eta1 && r_v<=r_l
            Lam=V; R_Lam=R_V;  RE=pDn*R_Lam(:); X_Lam_prod=X_V_prod; r_l=r_v;
            if pho>=eta2
                lambda=(lambda0+lambda)/2;
            else
                lambda=min(2*lambda,10^5);
            end
        else
            lambda=min(lambda*4,10^5);
        end
    end
    
    if inner_iter>20 % return if inner loop > 100
        stop_flag=1;
        break;
    end
    %  En(j)=r_l;
    inner_iter=inner_iter+1;
end
end

function V = linop(Block_d,x,n,t,reg)
V = zeros(size(x));

for i = 1:n
    V(:,i) = Block_d{i}*x(:,i);
end
V = 2*t*(V+V') + reg*x;
end