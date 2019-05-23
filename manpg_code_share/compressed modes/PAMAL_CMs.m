function [X_pamal, F_pamal,sparsity_pamal,time_PAMAL, error_XPQ,iter_pamal,flag_succ]=PAMAL_CMs(H,option,V)
% min Tr(X'*A*X)+ mu*norm(X,1) s.t. X'*X=Ir.
% A = -H
tic;
r=option.r;  n=option.n;
mu = option.mu;
maxiter =option.maxiter;
tol = option.tol;
mu2 = 1/mu;
A = - H;

% r = 2*(svds(B,1))^2+ 0.5*d;% ADMM stepsize
norminf=@(z) max(max(abs(z)));
lam_min=-1e2; lam_max=1e2;
c1=0.5; c2=c1; c3=c1;
%rng(40);
%beta = V + r/5; %stepsize
%beta = n*r*mu/1000 + 1;
beta_0 = r/20 + 1; 
beta = beta_0*2;
%beta = r*mu/2 + 1;

%[U,~]=eigs(A);  [~,r]=size(U);   X0= U(:,(r-n+1:r));
X0 = option.phi_init;
X_old = X0;
P_old = X0;
Q_old = X0;
lambda1 = zeros(n,r);
lambda2 =lambda1;
lambda1_old = lambda1;
lambda2_old  =lambda2;
R1_old = 0; R2_old=0;
F_PAMAL = zeros(maxiter,1);
%Z1 = 2*A +c1*eye(d);
dx = option.L/n;    %    r = 4/dx^2.*(sin(pi/4))^2 + n/2;
LAM_A = - (cos(2*pi*(0:n-1)'/n)-1)/dx^2;
FFTZ1 = 2*LAM_A +  c1;
flag = 0;
flag_time_out = 0;
not_full_rank = 0;
pamal_time = 0;
pamal_time = pamal_time + toc;
for iterp=1:maxiter
    tic;
    num(iterp)=1;
    % Z= Z1 + 2*r*eye(d) ;
    FFTZ = FFTZ1 + 2*beta;
    % X=Z\(lambda1_old+lambda2_old+r*Q_old+r*P_old+c1*X_old);
    X = real(ifft(bsxfun(@rdivide,fft( lambda1_old+lambda2_old+ beta*Q_old+ beta*P_old+c1*X_old ),FFTZ)));
    
    Q_m= (beta*X-lambda1_old+c2*Q_old)/(beta+c2);
    eta= 1/(mu2*(beta+c2));
    Q=sign(Q_m).* max(0, abs(Q_m)-eta);
    P_m = (beta*X+c3*P_old-lambda2_old)/(beta+c3);
    
    [U,~,V] = svd(P_m,0);
    P = U*V';
    %%%%%%%%%%%%%   svd P_m'*P_m
    %     [U, D, S] = svd(P_m'*P_m);
    %     D = diag(D);
    %     if abs(prod(D))>0
    %         P = P_m*U*diag(sqrt(1./D))*S';
    %     else
    %         not_full_rank = not_full_rank +1;
    %     end
    
    Theta1=beta*(Q_old-Q+P_old-P)+c1*(X_old-X);
    Theta2=c2*(Q_old-Q);
    Theta3=c3*(P_old-P);
    Theta=[Theta1,Theta2,Theta3];
    X_old = X; P_old = P; Q_old = Q;
    
    while norminf(Theta)> max(1e-6, (0.995)^iterp)    % Ji-paper 0.999: not good enough to get exact solution
        if num(iterp)> 100
            break;
        end
        %Z=2*A+(r+r+c1)*eye(d);
        %  X=Z\(lambda1_old+lambda2_old+r*Q_old+r*P_old+c1*X_old);
        X = real(ifft(bsxfun(@rdivide,fft( lambda1_old+lambda2_old+beta*Q_old+beta*P_old+c1*X_old ),FFTZ)));
        Q_m = (beta*X-lambda1_old+c2*Q_old)/(beta+c2);
        eta= 1/(mu2*(beta+c2));
        Q = sign(Q_m).* max(0, abs(Q_m)-eta);
        P_m = (beta*X+c3*P_old-lambda2_old)/(beta+c3);
        
        [U,~,V]=svd(P_m,0);
        P = U*V';
        %         [U, D, S] = svd(P_m'*P_m);
        %         D = diag(D);
        %         if abs(prod(D))>0
        %             P = P_m*(U*diag(sqrt(1./D))*S');
        %         end
        Theta1 = beta*(Q_old-Q+P_old-P)+c1*(X_old-X);
        Theta2 = c2*(Q_old-Q);
        Theta3 = c3*(P_old-P);
        Theta = [Theta1,Theta2,Theta3];
        
        X_old = X; P_old = P; Q_old = Q;
        num(iterp) = 1+num(iterp);
        
    end
    lambda1 =lambda1_old +beta*(Q-X);  lambda2=lambda2_old+beta*(P-X);
    lambda1_old= max(lam_min,lambda1);  lambda1_old= min(lam_max,lambda1_old);
    lambda2_old= max(lam_min,lambda2);  lambda2_old= min(lam_max,lambda2_old);
    R1 = Q-X;
    R2 = P-X;
    
    if norminf(R1)>norminf(R1_old)*0.99 || norminf(R2)>norminf(R2_old)*0.99
        %beta = min(beta_0*1000, beta*1.001);
        beta = beta*1.001;
    end
    %AP=-B'*(B*P);
    
    %AP = real(fft(bsxfun(@times,ifft( P ),LAM_A)));
    R1_old = R1; R2_old = R2;
    if iterp > 2
        % if abs(F_PALM(iterp)-F_PALM(iterp-1))/(abs(F_PALM(iterp))+1)<tol
        
        normXQ = norm(R1,'fro');
        normQ = norm(Q,'fro');
        normX = norm(X,'fro');
        normP = r;
        normXP = norm(R2,'fro');
        if   normXQ/max(1,max(normQ,normX)) + normXP/max(1,max(normP,normX)) <tol
            AP = A*P;
            F_PAMAL(iterp) = sum(sum(P.*(AP))) + mu*sum(sum(abs(P)));
            if F_PAMAL(iterp)<=option.F_manpg+ 1e-7
                break;
            end
            if norminf(Theta) < 2e-6 && normXQ/max(1,max(normQ,normX)) + normXP/max(1,max(normP,normX)) < 1e-7
                flag = 1; %different point
                break;
            end
        end       
       
    end
    pamal_time = pamal_time + toc;
    if pamal_time > r*60 % not over r minutes
        flag_time_out = 1;
        break;   
    end
end
P((abs(P)<=1e-5))=0;
X_pamal =P;
time_PAMAL = pamal_time;
inner_iterate = sum(num)/iterp;
error_XPQ = norm(X-P,'fro') + norm(X-Q,'fro');
sparsity_pamal= sum(sum(P==0))/(n*r);
X_manpg = option.X_manpg  ;
if iterp == maxiter || flag_time_out == 1
    fprintf('PAMAL  fails to converge in %d iterations \n', maxiter);
    %fprintf('PAMAL:Iter ***  Fval *** CPU  **** sparsity ***        iaverge_No.   ** err ***   inner_opt  \n');
    
    %print_format = ' %i     %1.5e    %1.2f       %1.2f                %2.2f         %1.3e    %1.3e   \n';
    %fprintf(1,print_format, iterp,F_PAMAL(iterp),time_PAMAL, sparsity_pamal, inner_iterate, error_XPQ,norminf(Theta));
    flag_succ = 0;
    F_pamal = 0;
    sparsity_pamal = 0;
    iter_pamal = 0;
    time_PAMAL = 0;
    
    
else
    if flag == 1 || norm(X_manpg*X_manpg'- X_pamal*X_pamal','fro')^2 > 0.1
        fprintf('PAMAL returns different point or fail to converge \n');
        
        flag_succ = 2; % different point
        fprintf('PAMAL:Iter ***  Fval *** CPU  **** sparsity ***        iaverge_No.   ** err ***   inner_opt  \n');
        
        print_format = ' %i     %1.5e    %1.2f       %1.2f                %2.2f         %1.3e    %1.3e   \n';
        fprintf(1,print_format, iterp,F_PAMAL(iterp),time_PAMAL, sparsity_pamal, inner_iterate, error_XPQ,norminf(Theta));
        F_pamal = 0;
        sparsity_pamal = 0;
        iter_pamal = 0;
        time_PAMAL = 0;
    else
        flag_succ = 1;
        
        F_pamal = F_PAMAL(iterp);
        %Eigspalm=eig(P'*A*P);
        
        iter_pamal = iterp;
        % semilogy((1:iter),F(1:iter)-min(min(F),min(F2)));
        % hold on;
        % semilogy((1:i),F2(1:i)-min(min(F),min(F2)))
        fprintf('PAMAL:Iter ***  Fval *** CPU  **** sparsity ***        iaverge_No.   ** err ***   inner_opt  \n');
        
        print_format = ' %i     %1.5e    %1.2f       %1.2f                %2.2f         %1.3e    %1.3e   \n';
        fprintf(1,print_format, iterp,F_PAMAL(iterp),time_PAMAL, sparsity_pamal, inner_iterate, error_XPQ,norminf(Theta));
    end
end