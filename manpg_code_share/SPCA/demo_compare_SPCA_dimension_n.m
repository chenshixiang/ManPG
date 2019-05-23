%function compare_spca

clear;
close all;
addpath ../misc
addpath ../SSN_subproblem
n_set=[100; 200;  500; 800; 1000; 1500; ]; %dimension
%n_set = 2.^(6:9);
r_set = [1;2;5;10;];   % rank

mu_set = 0.8;
index = 1;
for id_n = 1:1%size(n_set,1)        % n  dimension
    
    n = n_set(id_n);
    fid =1;
    
    for id_r = 3%size(r_set,1) % r  number of column
        for id_mu = 1          % mu  sparse parameter
            r = r_set(id_r);
            %mu = mu_set(id_mu);
            
            succ_no_manpg = 0;  succ_no_manpg_BB = 0; succ_no_SOC = 0;  succ_no_PAMAL = 0; succ_no_sub = 0;
            diff_no_SOC = 0;  diff_no_PAMAL = 0;  diff_no_sub = 0;
            fail_no_SOC = 0;  fail_no_PAMAL = 0;  fail_no_sub = 0;
            for test_random = 1:2  %times average.
                fprintf(fid,'==============================================================================================\n');
                
                rng('shuffle');
                %rng(70);
                m = 50;
                B = randn(m,n);
                type = 0; % random data matrix
                if (type == 1) %covariance matrix
                    scale = max(diag(B)); % Sigma=A/scale;
                elseif (type == 0) %data matrix
                    B = B - repmat(mean(B,1),m,1);
                    %                     scale = [];
                    %                     for id = 1:n
                    %                         scale = [scale norm(B(:,id))];
                    %                     end
                    %                     scale = max(scale);
                    %                     B = B/scale;
                    B = normc(B);
                    %  Sigma=A'*A;
                end
                %B(abs(B)<0.1) = 0;
                mu = mu_set(id_mu);
                fprintf(fid,'- n -- r -- mu --------\n');
                fprintf(fid,'%4d %3d %3.3f \n',n,r,mu);
                fprintf(fid,'----------------------------------------------------------------------------------\n');
                
                rng('shuffle');
                %rng(177);
                [phi_init,~] = svd(randn(n,r),0);  % random intialization
                %[phi_init,~] = eigs(H,r);    % singular value initialization
                option_Rsub.F_manpg = -1e10;
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 5e2;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = type;
                
                [phi_init, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(B,option_Rsub);
                
                
                %%%%%  manpg parameter
                option_manpg.adap = 0;    option_manpg.type =type;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 20000;  option_manpg.tol =1e-8*n*r;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = mu;
                %option_manpg.L = L;
                %option_manpg.inner_tol =1e-11;
                option_manpg.inner_iter = 100;
                %%%%%% soc parameter
                option_soc.phi_init = phi_init; option_soc.maxiter = 20000;  option_soc.tol =1e-4;
                option_soc.r = r;    option_soc.n = n;  option_soc.mu=mu;
                %option_soc.L= L;
                option_soc.type = type;
                %%%%%% PAMAL parameter
                option_PAMAL.phi_init = phi_init; option_PAMAL.maxiter =20000;  option_PAMAL.tol =1e-4;
                %option_PAMAL.L = L;   option_PAMAL.V = V;
                option_PAMAL.r = r;   option_PAMAL.n = n;  option_PAMAL.mu=mu;   option_PAMAL.type = type;
                %    B = randn(d,d)+eye(d,d); B = -B'*B;
                [X_manpg, F_manpg(test_random),sparsity_manpg(test_random),time_manpg(test_random),...
                    maxit_att_manpg(test_random),succ_flag_manpg, lins(test_random),in_av(test_random)]= manpg_orth_sparse(B,option_manpg);
                if succ_flag_manpg == 1
                    succ_no_manpg = succ_no_manpg + 1;
                end
                
                option_manpg.F_manpg = F_manpg(test_random);
                [X_manpg_BB, F_manpg_BB(test_random),sparsity_manpg_BB(test_random),time_manpg_BB(test_random),...
                    maxit_att_manpg_BB(test_random),succ_flag_manpg_BB,lins_adap(test_random),in_av_adap(test_random)]= manpg_orth_sparse_adap(B,option_manpg);
                if succ_flag_manpg_BB == 1
                    succ_no_manpg_BB = succ_no_manpg_BB + 1;
                end
                
                %%%%%% Riemannian subgradient parameter
                option_Rsub.F_manpg = F_manpg(test_random);
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e1;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = type;
                
                [X_Rsub, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(B,option_Rsub);
                %phi_init = X_Rsub;
                if succ_flag_sub == 1
                    succ_no_sub = succ_no_sub + 1;
                end
                option_soc.F_manpg = F_manpg(test_random);
                option_soc.X_manpg = X_manpg;
                option_PAMAL.F_manpg = F_manpg(test_random);
                option_PAMAL.X_manpg = X_manpg;
                [X_Soc, F_soc(test_random),sparsity_soc(test_random),time_soc(test_random),...
                    soc_error_XPQ(test_random),maxit_att_soc(test_random),succ_flag_SOC]= soc_spca(B,option_soc);
                if succ_flag_SOC == 1
                    succ_no_SOC = succ_no_SOC + 1;
                end
                option_PAMAL.F_manpg = F_manpg(test_random);
                [X_pamal, F_pamal(test_random),sparsity_pamal(test_random),time_pamal(test_random),...
                    pam_error_XPQ(test_random), maxit_att_pamal(test_random),succ_flag_PAMAL]= PAMAL_spca(B,option_PAMAL);
                if succ_flag_PAMAL ==1
                    succ_no_PAMAL = succ_no_PAMAL + 1;
                end
                
                if succ_flag_sub == 0
                    fail_no_sub = fail_no_sub + 1;
                end
                if succ_flag_sub == 2
                    diff_no_sub = diff_no_sub + 1;
                end
                if succ_flag_SOC == 0
                    fail_no_SOC = fail_no_SOC + 1;
                end
                if succ_flag_SOC == 2
                    diff_no_SOC = diff_no_SOC + 1;
                end
                if succ_flag_PAMAL == 0
                    fail_no_PAMAL = fail_no_PAMAL + 1;
                end
                if succ_flag_PAMAL == 2
                    diff_no_PAMAL = diff_no_PAMAL + 1;
                end
                if succ_flag_manpg == 1
                    F_best(test_random) =  F_manpg(test_random);
                end
                if succ_flag_manpg_BB == 1
                    F_best(test_random) =  min( F_best(test_random), F_manpg_BB(test_random));
                end
                if succ_flag_SOC == 1
                    F_best(test_random) =  min( F_best(test_random), F_soc(test_random));
                end
                if succ_flag_PAMAL == 1
                    F_best(test_random) =  min( F_best(test_random), F_pamal(test_random));
                end
                
            end
            
        end
        
    end
    Result(index,1) = sum(lins)/succ_no_manpg;  Result(index,2) = sum(in_av)/succ_no_manpg;
    Result(index,3) = sum(lins_adap)/succ_no_manpg;  Result(index,4) = sum(in_av_adap)/succ_no_manpg;
    
    %Result(index,5) = succ_no_manpg_BB;  Result(index,6) = succ_no_SOC;   Result(index,7)=succ_no_PAMAL; Result(index,8)=succ_no_sub;
    Result(index,5) = succ_no_manpg_BB;  Result(index,6) = succ_no_SOC;  Result(index,7) = fail_no_SOC;   Result(index,8) = diff_no_SOC;
    Result(index,9)= succ_no_PAMAL;      Result(index,10)=fail_no_PAMAL;  Result(index,11)= diff_no_PAMAL;
    index = index +1;
    
    iter.manpg(id_n) =  sum(maxit_att_manpg)/succ_no_manpg;
    iter.manpg_BB(id_n) =  sum(maxit_att_manpg_BB)/succ_no_manpg_BB;
    iter.soc(id_n) =  sum(maxit_att_soc)/succ_no_SOC;
    iter.pamal(id_n) =  sum(maxit_att_pamal)/succ_no_PAMAL;
    iter.Rsub(id_n) =  sum(maxit_att_Rsub)/succ_no_sub;
    
    time.manpg(id_n) =  sum(time_manpg)/succ_no_manpg;
    time.manpg_BB(id_n) =  sum(time_manpg_BB)/succ_no_manpg_BB;
    time.soc(id_n) =  sum(time_soc)/succ_no_SOC;
    time.pamal(id_n) =  sum(time_pamal)/succ_no_PAMAL;
    time.Rsub(id_n) =  sum(time_Rsub)/succ_no_sub;
    
    Fval.manpg(id_n) =  sum(F_manpg)/succ_no_manpg;
    Fval.manpg_BB(id_n) =  sum(F_manpg_BB)/succ_no_manpg_BB;
    Fval.soc(id_n) =  sum(F_soc)/succ_no_SOC;
    Fval.pamal(id_n) =  sum(F_pamal)/succ_no_PAMAL;
    Fval.Rsub(id_n) =  sum(F_Rsub)/succ_no_sub;
    Fval.best(id_n) = sum(F_best)/succ_no_manpg;
    
    Sp.manpg(id_n) =  sum(sparsity_manpg)/succ_no_manpg;
    Sp.manpg_BB(id_n) =  sum(sparsity_manpg_BB)/succ_no_manpg_BB;
    Sp.soc(id_n) =  sum(sparsity_soc)/succ_no_SOC;
    Sp.pamal(id_n) =  sum(sparsity_pamal)/succ_no_PAMAL;
    Sp.Rsub(id_n) =  sum(sparsity_Rsub)/succ_no_sub;
    fprintf(fid,'==============================================================================================\n');
    
    fprintf(fid,' Alg ****        Iter *****  Fval *** sparsity ** cpu *** Error ***\n');
    
    print_format =  'ManPG:      %1.3e  %1.5e    %1.2f      %3.2f \n';
    fprintf(fid,print_format, iter.manpg(id_n), Fval.manpg(id_n), Sp.manpg(id_n),time.manpg(id_n));
    print_format =  'ManPG_adap: %1.3e  %1.5e    %1.2f      %3.2f \n';
    fprintf(fid,print_format, iter.manpg_BB(id_n), Fval.manpg_BB(id_n), Sp.manpg_BB(id_n),time.manpg_BB(id_n));
    print_format =  'SOC:        %1.3e  %1.5e    %1.2f      %3.2f    %1.3e\n';
    fprintf(fid,print_format,iter.soc(id_n) , Fval.soc(id_n), Sp.soc(id_n) ,time.soc(id_n),mean(soc_error_XPQ));
    print_format =  'PAMAL:      %1.3e  %1.5e    %1.2f      %3.2f    %1.3e \n';
    fprintf(fid,print_format,iter.pamal(id_n) ,  Fval.pamal(id_n) ,Sp.pamal(id_n),time.pamal(id_n),mean(pam_error_XPQ));
    print_format =  'Rsub:       %1.3e  %1.5e    %1.2f      %3.2f  \n';
    fprintf(fid,print_format,iter.Rsub(id_n) ,  Fval.Rsub(id_n) ,Sp.Rsub(id_n),time.Rsub(id_n));
end

%% plot

% figure(1);
% plot(n_set, time.manpg, 'r-','linewidth',1); hold on;
% plot(n_set, time.manpg_BB, 'k-','linewidth',1); hold on;
% plot(n_set, time.soc, 'b--','linewidth',1); hold on;
% plot(n_set, time.pamal, 'c-.','linewidth',2); hold on;
% %plot(n_set, time.Rsub, 'g-.','linewidth',2);
% xlabel('dimenion-n');   ylabel('CPU');
% title(['comparison on CPU: different dimension',',r=',num2str(r),',\mu=',num2str(mu)]);
% legend('ManPG','ManPG-adap','SOC','PAMAL','Location','NorthWest');
% filename_pic1 = ['SPCA_CPU_n',  '_' num2str(r) '_' num2str(mu)  '.eps'];
% saveas(gcf,filename_pic1,'epsc')
%
%
% figure(2)
% semilogy(n_set, Fval.manpg-Fval.best+1e-16, 'r-s','MarkerSize',10,'linewidth',1); hold on;
% semilogy(n_set, Fval.manpg_BB-Fval.best+1e-16, 'k-o','MarkerSize',6,'linewidth',1); hold on;
% semilogy(n_set, Fval.soc-Fval.best+1e-16, 'b-d','MarkerSize',8,'linewidth',1); hold on;
% semilogy(n_set, Fval.pamal-Fval.best+1e-16, 'c-.','MarkerSize',20,'linewidth',1.5); hold on;
% %semilogy(n_set, Fval.Rsub -Fval.best+1e-16, 'g-.','MarkerSize',20,'linewidth',1.5);
% %legend('ManPG','ManPG-BB','SOC','PAMAL','Rsub','Location','NorthWest');
% legend('ManPG','ManPG-adap','SOC','PAMAL','Location','NorthWest');
%
% xlabel('dimenion-n');   ylabel('fucntion value difference');
% title(['comparison on function value difference: different dimension',',r=',num2str(r),',\mu=',num2str(mu)]);
% filename_pic2 = ['SPCA_Fval_n',  '_' num2str(r) '_' num2str(mu)  '.eps'];
% saveas(gcf,filename_pic2,'epsc')
%
% %
% %
% figure(3)
% plot(n_set, Sp.manpg, 'r-s','MarkerSize',10,'linewidth',1); hold on;
% plot(n_set, Sp.manpg_BB, 'k-o','MarkerSize',6,'linewidth',1); hold on;
% plot(n_set, Sp.soc, 'b-d','MarkerSize',8,'linewidth',1); hold on;
% plot(n_set, Sp.pamal, 'c-.','MarkerSize',20,'linewidth',1.5); hold on;
% %plot(n_set, Sp.Rsub, 'g-.','MarkerSize',20,'linewidth',1.5); %hold on;
% xlabel('dimenion-n');   ylabel('sparsity');
% legend('ManPG','ManPG-adap','SOC','PAMAL','Location','SouthEast');
% title(['comparison on sparsity: different dimension',',r=',num2str(r),',\mu=',num2str(mu)]);
% filename_pic3 = ['SPCA_Sparsity_n',  '_' num2str(r)  '_' num2str(mu) '.eps'];
% saveas(gcf,filename_pic3,'epsc')
%
%
% figure(4)
% plot(n_set, iter.manpg, 'r-s','MarkerSize',10,'linewidth',1); hold on;
% plot(n_set, iter.manpg_BB, 'k-o','MarkerSize',6,'linewidth',1); hold on;
% plot(n_set, iter.soc, 'b-d','MarkerSize',8,'linewidth',1); hold on;
% plot(n_set, iter.pamal, 'c-.','MarkerSize',20,'linewidth',1.5); %hold on;
% %plot(n_set, iter.Rsub, 'g-.','MarkerSize',20,'linewidth',1.5); %hold on;
% xlabel('dimenion-n');   ylabel('iter');
% legend('ManPG','ManPG-adap','SOC','PAMAL','Location','NorthWest');
% title(['comparison on iter: different dimension',',r=',num2str(r),',\mu=',num2str(mu)]);
% filename_pic4= ['SPCA_iter_n',  '_'  num2str(r)  '_' num2str(mu) '.eps'];
% saveas(gcf,filename_pic4,'epsc')
%
% close(figure(1));
% close(figure(2));
% close(figure(3));
% close(figure(4));
% filename = ['SPCA_comparison_n_'  num2str(r_set(id_r)) '_'  num2str(mu_set(id_mu)) '.csv'];
% csvwrite( filename, Result);



