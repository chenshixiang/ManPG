function [X_Soc, F_SOC,sparsity_soc,time_soc,error_XPQ, iter_soc, flag_succ]=soc_spca(B,option)
%min -Tr(X'*A*X)+ mu*norm(X,1) s.t. X'*X=Ir.
% A = B'*B type = 0 or A = B  type = 1
tic;
r = option.r;
n = option.n;
mu = option.mu;
maxiter =option.maxiter;
tol = option.tol;
type = option.type;
if type==0 % data matrix
    A = -B'*B;
else
    A = -B;
end

h=@(X) mu*sum(sum(abs(X)));
%rho = svds(B,1)^2 + r/2;%  stepsize
%rho = svds(B,1)^2 + n*r*mu/25 + 1;
%rho = svds(B,1)^2 + n/50 ;% good for mu and r
rho = 2* svds(B,1)^2  ;%  n/30 not converge   1.9* sometimes not converge
lambda = rho;
P = option.phi_init;    Q = P;
Z = zeros(n,r); 
b=Z;  
F_ad=zeros(maxiter,1);
not_full_rank = 0;

%chA = chol( 2*A + (r+lambda)*eye(d));
Ainv = inv( 2*A + (rho+lambda)*eye(n));
flag_maxiter = 0;

for itera=1:maxiter
    LZ = rho*(P-Z)+lambda*(Q-b);
    %   X=A_bar\LB;
    %  X = chA\(chA'\LZ);
    X = Ainv*LZ;
    %%%% shrinkage Q
    Q = sign(X+b).*max(0,abs(X+b)-mu/lambda);
    
    %%%% solve P
    
    Y = X + Z;
    %%%%%%%%%%%%%   svd Y'*Y
    %     [U, D, S] = svd(Y'*Y);
    %     D = diag(D);
    %     if abs(prod(D))>0
    %         P = Y*(U*diag(sqrt(1./D))*S');
    %     else
    %         not_full_rank = not_full_rank+1;
    %     end
    [U,~,V]= svd(Y,0);
    P = U*V';
    %%%%%%%%%
    Z  = Z+X-P;
    b  = b+X-Q;
    
    if itera>2
        normXQ = norm(X-Q,'fro');
        normQ = norm(Q,'fro');
        normX = norm(X,'fro');
        normP = r;
        normXP = norm(X-P,'fro');
        if  normXQ/max(1,max(normQ,normX)) + normXP/max(1,max(normP,normX)) <tol
            if type == 0 % data matrix
                AP = -(B'*(B*P));
            else
                AP = -(B*P);
            end
            F_ad(itera)= sum(sum(X.*(AP)))+h(P);
            if F_ad(itera)<= option.F_manpg+ 1e-7
                break;
            end
        end
        %         if   normXQ  + normXP <1e-9*r
        %             break;
        %         end
    end
    
    P_old=P;
    if itera ==maxiter
        flag_maxiter =1;
    end
end
P((abs(P)<=1e-5))=0;
X_Soc=P;
time_soc= toc;
error_XPQ = norm(X-P,'fro') + norm(X-Q,'fro');
X_manpg = option.X_manpg;
sparsity_soc= sum(sum(P==0))/(n*r);
if itera == maxiter
    flag_succ = 0; %fail
    F_SOC = 0;
    sparsity_soc = 0;
    iter_soc = 0;
    fprintf('SOC fails to converge  \n');
    
    %fprintf('Soc:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    
    %print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    %fprintf(1,print_format, itera, F_ad(itera), time_soc, sparsity_soc,  error_XPQ);
    time_soc = 0;
else
    if norm(X_manpg*X_manpg'- X_Soc*X_Soc','fro')^2 > 0.1
        fprintf('SOC returns different point \n');
        fprintf('Soc:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
        
        print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
        fprintf(1,print_format, itera, F_ad(itera), time_soc, sparsity_soc,  error_XPQ);
        flag_succ = 2; % different point
        F_SOC = 0;
        sparsity_soc = 0;
        iter_soc = 0;
        
        time_soc = 0;
    else
        
        flag_succ = 1; % success
        F_SOC = F_ad(itera);
        iter_soc = itera;
        % residual_Q = norm(Q'*Q-eye(n),'fro')^2;
        fprintf('Soc:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
        
        print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
        fprintf(1,print_format, itera, F_ad(itera), time_soc, sparsity_soc,  error_XPQ);
    end
end
