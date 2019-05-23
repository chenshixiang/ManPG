function [X_manpg, F_manpg,sparsity,time_manpg,iter_adap,flag_succ,num_linesearch,mean_ssn]  = manpg_orth_sparse_adap(B,option)
%min -Tr(X'*A*X)+ mu*norm(X,1) s.t. X'*X=Ir.
% A = B'*B type = 0 or A = B  type = 1
% mu can be a vector with weighted parameter
%parameters
tic;
r = option.r;%number of col
n = option.n;%dim
mu = option.mu;
maxiter = option.maxiter;
tol = option.tol;
h = @(X) sum(mu.*sum(abs(X)));
inner_iter = option.inner_iter;
prox_fun = @(b,lambda,r) proximal_l1(b,lambda,r);
inner_flag = 0;
%setduplicat_pduplicat(r);
Dn = sparse(DuplicationM(r));
pDn = (Dn'*Dn)\Dn';
type = option.type;
t_min = 1e-4; % minimum stepsize
%%
%initial point
X = option.phi_init;
if type == 1
    L = 2*abs(eigs(full(B),1));
    %  L=2*abs(eigs(B,1));
else
    L = 2*(svds(full(B),1))^2;
end

if type == 1
    AX = B*X;
else
    AX = B'*(B*X);
end
F(1) = -sum(sum(X.*(AX)))+h(X);
num_inner = zeros(maxiter,1);
opt_sub = num_inner;
num_linesearch = 0;
num_inexact = 0;
alpha =1;
t0 = 1/L;
t = t0;
%inner_tol  = 0.1*tol^2*t^2;
linesearch_flag = 0;
for iter = 2:maxiter
    %   fprintf('manpg__________________iter: %3d, \n', iter);
    ngx = 2*AX; % negative gradient       pgx=gx-X*xgx;  %projected gradient
    neg_pgx = ngx; % grad or projected gradient both okay
    %% subproblem
    if alpha < t_min || num_inexact > 10
        inner_tol = max(1e-15, min(1e-13,1e-5*tol*t^2)); % subproblem inexact;
    else
        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
    end
    
    if iter == 2
        [ PY,num_inner(iter),Lam, opt_sub(iter),in_flag] = Semi_newton_matrix(n,r,X,t,X + t*neg_pgx,mu*t,inner_tol,prox_fun,inner_iter,zeros(r),Dn,pDn);
        %      [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    else
         [ PY,num_inner(iter),Lam, opt_sub(iter),in_flag] = Semi_newton_matrix(n,r,X,t,X + t*neg_pgx,mu*t,inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
        %     [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    end
    
    %     if iter ==2
    %         [Y,~, dual_y,dual_z, ~,  ~, ~, ~, subiter, ls, cg_iter] =SSNAL_mat(eye(n),eye(n),X-t*pgx, X, eye(n), eye(n),X, X,zeros(r),zeros(n,r), eye(r), mu*t, 50, 1e-4) ;
    %     else
    %         [Y,~, dual_y,dual_z, ~,  ~, ~, ~, subiter, ls, cg_iter] = SSNAL_mat(eye(n),eye(n),X-t*pgx, X, eye(n),eye(n),X, X, dual_y, dual_z, eye(r), mu*t, 50, 1e-3) ;
    %     end
    if in_flag == 1   % subprolem not exact.
        inner_flag = 1 + inner_flag;
    end
    alpha = 1;
    D = PY-X; %descent direction D
    
    [U, SIGMA, S] = svd(PY'*PY);   SIGMA =diag(SIGMA);    Z = PY*(U*diag(sqrt(1./SIGMA))*S');
    % [Z,R]=qr(PY,0);       Z = Z*diag(sign(diag(R))); %old version need consider the sign
    
    if type == 1
        AZ = B*Z;
    else
        AZ = B'*(B*Z);
    end
    %   AZ = real(ifft( LAM_manpg.*fft(Z) ));
    
    f_trial = -sum(sum(Z.*(AZ)));
    F_trial = f_trial+h(Z);   normDsquared=norm(D,'fro')^2;

    %% linesearch
    while F_trial >= F(iter-1)-0.5/t*alpha*normDsquared
        alpha = 0.5*alpha;
        linesearch_flag = 1;
        num_linesearch = num_linesearch+1;
        if alpha<t_min
            num_inexact = num_inexact + 1;
            break;
        end
        PY = X+alpha*D;
        %  [U,~,V]=svd(PY,0);  Z=U*V';
        %  [Z,R]=qr(PY,0);   Z = Z*diag(sign(diag(R)));  %old version need consider the sign
        [U, SIGMA, S] = svd(PY'*PY);   SIGMA =diag(SIGMA);   Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        if type ==1
            AZ= B*Z;
        else
            AZ = B'*(B*Z);
        end
        %  flag_linesearch(iter) = 1+flag_linesearch(iter); % linesearch flag
        f_trial = -sum(sum(Z.*(AZ)));
        F_trial = f_trial+ h(Z);
    end
    X = Z; AX = AZ;
    F(iter) = F_trial;
   
    if F_trial < option.F_manpg + 1e-7
        break;
    end
    if linesearch_flag == 0
        t = t*1.01;
    else
        t = max(t0,t/1.01);
    end
    
    linesearch_flag = 0;
    
end
X((abs(X)<=1e-5))=0;
X_manpg = X;
time_manpg = toc;
mean_ssn = sum(num_inner)/(iter-1);

if iter == maxiter %&& sqrt(normDsquared)/t > 1e-1
    flag_succ = 0;
    sparsity  = 0;
    F_manpg = 0;
    time_manpg = 0;
    iter_adap = 0;
else
    flag_succ = 1;
    iter_adap = iter;
    sparsity= sum(sum(X_manpg==0))/(n*r);
    F_manpg = F(iter);
    
    fprintf('ManPG_adap:Iter ***  Fval *** CPU  **** sparsity ***inner_inexact&averge_No. ** opt_norm ** total_linsea \n');
    print_format = ' %i     %1.5e    %1.2f     %1.2f         %4i   %2.2f                %1.3e        %d \n';
    fprintf(1,print_format, iter-1,min(F), time_manpg,sparsity, inner_flag, sum(num_inner)/(iter-1) ,sqrt(normDsquared)/t,num_linesearch);
end