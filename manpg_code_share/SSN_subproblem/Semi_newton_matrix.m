function [Z,j,Lam,r_l,stop_flag] = Semi_newton_matrix(n,r,X,t, B,mut,inner_tol,prox_fun,inner_max_iter,Lam0,Dn,pDn)
%find the zero of X'*Z(Lam)+ Z(Lam)'*X=2*I, where z(Lam) is the proximal
%point of B+2*t*X*Lam      %prox_number=0;
XtX = eye(r);
Xt  = X';
%global  Dn pDn

stop_flag = 0;
Lam = Lam0;
X_Lam_prod = B + 2*t*(X*Lam);
[Z,Act_set,Inact_set] = prox_fun(X_Lam_prod,mut,r);
ZX = Z'*X;
R_Lam = ZX+ZX'-2*eye(r);
RE = pDn*R_Lam(:);
%  s = s - r_s/gd;
r_l = norm(R_Lam,'fro');
g=zeros(r^2);
%g=sparse(n^2,n^2);
lambda = 0.2;
eta1 = 0.1;
eta2 = 0.3;
sigma = r_l;
lambda0 = 1e-3;
j = 0;
while  r_l^2 > inner_tol
    
    reg = lambda*max(min(r_l,0.1),1e-11);
    nnzZ = nnz(Z);
    if r < 15
        if nnzZ > floor(r*(r+1)/2) % directly compute inverse of G
            for i = 1:r
                g((i*r-r+1:i*r),(i*r-r+1:i*r)) = X'*(Act_set(:,i).*X);
            end
            G = 4*t*(pDn*(g*Dn));
            [LG,UG] = lu(G+ reg*eye(r*(r+1)/2));
            new_d = -(UG\(LG\RE));   % use lu decompostion  solve linear system
            %new_d1 = -(G+ reg*eye(r*(r+1)/2))\RE;
            %LG = chol(G+ reg*eye(r*(r+1)/2),'lower'); %new_d = LG'\(LG\(-RE));
        else% Use SWM Formula  dim: number of non zero of Z
            Xstack = zeros(nnzZ,r^2);
            dim  =0;
            for i = 1:r
                row = find(Act_set(:,i)==1);
                Xstack((dim+1:dim+length(row)) ,(i*r-r+1:i*r)) =  X(row,:);
                dim = length(row) + dim;
            end
            V = Xstack*Dn;   U = 4*t* (pDn*Xstack');
            [LG,UG] = lu(eye(nnzZ) + 1/reg*(V*U));
            new_d = -(1/reg*RE - 1/reg^2 *U*(UG\(LG\(V*RE))));
            % LG = chol(eye(nnzZ) + 1/reg*(V*U),'lower');
            % new_d =  -(1/reg*RE - 1/reg^2 *U*(LG'\(LG\(V*RE))));
            %new_d = -(1/reg*RE - 1/reg^2 *U*((eye(nnzZ) + 1/reg*V*U)\(V*RE)));
        end
        new_d = Dn*new_d;
        %norm_d = norm(new_d)^2;
        new_d = reshape(new_d,r,r);
    else% implement cg
        Blkd = cell(r,1);
        for i=1:r
            % Blkd{i} = X'*(Act_set(:,i).*X);
            ind  =Act_set(:,i);
            if sum(ind) < n/2
                X_ind = X(ind,:);
                Blkd{i} = X_ind'*X_ind;
            else
                ind = Inact_set(:,i);
                % X_ind = X(ind,:);
                X_ind = Xt(:,ind);
                Blkd{i} = XtX -  X_ind*X_ind';
            end
        end
        fun =@(x) linop(Blkd,x,r,t,reg);
        [new_d,cg_iter,res] = conjgrad(fun,-R_Lam,min(1e-4,1e-3*r_l));
        % norm_d = norm(new_d)^2;
    end
     t_new = 1;
    X_d_prod = 2*t*(X*new_d);
    X_Lam_new_prod = X_Lam_prod + t_new*X_d_prod;
   
    %Lam_new = Lam + new_d ;
    [Z, Act_set,Inact_set] = prox_fun(X_Lam_new_prod,mut,r);
    ZX=Z'*X;
    R_Lam_new = ZX+ZX'-2*eye(r);
    
    r_l_new = norm(R_Lam_new,'fro');
    while  r_l_new ^2 >=  r_l^2 *(1-  0.001 *t_new) && t_new > 1e-3
        t_new = 0.5*t_new;
        %Lam_new = Lam + t_new* new_d ;
        X_Lam_new_prod = X_Lam_prod + t_new*X_d_prod;
        [Z, Act_set,Inact_set] = prox_fun(X_Lam_new_prod,mut,r);
        ZX = Z'*X;
        R_Lam_new = ZX+ZX'-2*eye(r);
        r_l_new = norm(R_Lam_new,'fro');
        %pho = -sum(sum(R_Lam_new.*new_d))/norm_d;
        %      if r_l_new <= sigma % New step
       
    end
        Lam = Lam + t_new* new_d ;
        r_l = r_l_new;
    %   sigma = r_l;
        R_Lam = R_Lam_new;
        RE = pDn*R_Lam(:);
        X_Lam_prod = X_Lam_new_prod;
    %     else
    %         V = Lam - sum(sum(R_Lam_new.*(Lam-Lam_new)))/r_l_new^2* R_Lam_new;
    %         X_V_prod = B+2*t*(X*V);
    %         [Z,Act_set,Inact_set] = prox_fun(X_V_prod,mut,r);
    %         ZX = Z'*X;
    %         R_V = ZX+ZX'-2*eye(r);
    %         r_v = norm(R_V,'fro');
    %
    %         if pho>=eta1 && r_v<=r_l  %projection step
    %             Lam = V;
    %             R_Lam = R_V;
    %             RE = pDn*R_Lam(:);
    %             X_Lam_prod = X_V_prod;
    %             r_l = r_v;
    %         else
    %             if pho>=eta1 && r_v > r_l % contraction step
    %                 Lam = Lam - 2*t*R_Lam;
    %                 X_Lam_prod = B + 2*t*(X*Lam);
    %                 [Z,Act_set,Inact_set] = prox_fun(X_Lam_prod,mut,r);
    %                 ZX = Z'*X;
    %                 R_Lam = ZX+ZX'-2*eye(r);
    %                 RE = pDn*R_Lam(:);
    %                 r_l = norm(R_Lam,'fro');
    %             end
    %         end
    %     end
    %
    %     if pho >= eta2
    %         lambda=(lambda0+lambda)/2;
    %     else
    %         if pho> eta1
    %             lambda = min(2*lambda,10^4);
    %
    %         else
    %             lambda = min(lambda*4,10^4);
    %         end
    %     end
    
    if j>inner_max_iter % return if inner loop > 100
        stop_flag = 1;
        break;
    end
    %  En(j)=r_l;
    j = j+1;
end
end

function V = linop(Block_d,x,n,t,reg)
V = zeros(size(x));

for i = 1:n
    V(:,i) = Block_d{i}*x(:,i);
end
V = 2*t*(V+V') + reg*x;
end