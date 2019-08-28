%function compare_spca

clear all;
close all;
addpath ../misc
addpath ../SSN_subproblem
%n_set=[100; 500; 1000; 1500; 2000]; %dimension
n_set = 128;
r_set = 15;   % rank
mu_set = 1/30;
index = 1;
for id_n =  1:size(n_set,2)        % n  dimension
    n = n_set(id_n);
    fid =1;
    for id_r = 1%size(r_set,1) % r number of column
        
        for id_mu = 1%mu
            r = r_set(id_r);
            %mu = mu_set(id_mu);
            fprintf(fid,'===================================\n');
            succ_no_manpg = 0;  succ_no_manpg_BB = 0; succ_no_SOC = 0;  succ_no_PAMAL = 0; succ_no_sub = 0;
            diff_no_SOC = 0;  diff_no_PAMAL = 0;  diff_no_sub = 0;
            fail_no_SOC = 0;  fail_no_PAMAL = 0;  fail_no_sub = 0;
            
            for test_random =1:1%times average.
                L = 50; dx = L/n;  V = 0;
                %  phi_init = CMs_1D_initial(d,n,dx); % initial point ---Guass kernel
                H = -Sch_matrix(0,L,n); %  schrodinger operator
                %H = H/abs(eigs(H,1));
                mu = mu_set(id_mu);
                fprintf(fid,'- n -- r -- mu --------\n');
                fprintf(fid,'%4d %3d %3.3f \n',n,r,mu);
                rng('shuffle');
                %rng(10);
                [phi_init,~] = svd(randn(n,r),0);  % random intialization
                %[phi_init,~] = eigs(H,r);    % singular value initialization
                option_Rsub.F_manpg = -1e10;
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = n*r;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = 1;
                
                [phi_init, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(H,option_Rsub);
                
                
                %%%%%  manpg parameter
                %option_manpg.adap = 0;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 30000;  option_manpg.tol =1e-8*n*r;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = mu;
                option_manpg.L = L;
                %option_manpg.inner_tol =1e-11;
                option_manpg.inner_iter = 100;
                %%%%%% soc parameter
                option_soc.phi_init = phi_init; option_soc.maxiter = 30000;  option_soc.tol =1e-4;
                option_soc.r = r;    option_soc.n = n;  option_soc.mu=mu;
                option_soc.L= L;
                %%%%%% PAMAL parameter
                option_PAMAL.phi_init = phi_init; option_PAMAL.maxiter = 30000;  option_PAMAL.tol =1e-4;
                option_PAMAL.L = L;   option_PAMAL.V = V;
                option_PAMAL.r = r;   option_PAMAL.n = n;  option_PAMAL.mu=mu;
                %    B = randn(d,d)+eye(d,d); B = -B'*B;
                [X_manpg, F_manpg(test_random),sparsity_manpg(test_random),time_manpg(test_random),...
                    maxit_att_manpg(test_random),succ_flag_manpg,lins(test_random),in_av(test_random)]= manpg_CMS(H,option_manpg,dx,V);
                if succ_flag_manpg == 1
                    succ_no_manpg = succ_no_manpg + 1;
                end
                
                option_manpg.F_manpg = F_manpg(test_random);
                [X_manpg_adap, F_manpg_BB(test_random),sparsity_manpg_BB(test_random),time_manpg_BB(test_random),...
                    maxit_att_manpg_BB(test_random),succ_flag_manpg_BB,lins_adap(test_random),in_av_adap(test_random)]= manpg_CMS_adap(H,option_manpg,dx,V);
                if succ_flag_manpg_BB == 1
                    succ_no_manpg_BB = succ_no_manpg_BB + 1;
                end
                
                %%%%%% Riemannian subgradient parameter
%                 option_Rsub.F_manpg = F_manpg(test_random);
%                 option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e1;  option_Rsub.tol = 1e-3;
%                 option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = 1;
%                 
%                 [X_Rsub, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
%                     maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(H,option_Rsub);
%                 phi_init = X_Rsub;
%                 if succ_flag_sub == 1
%                     succ_no_sub = succ_no_sub + 1;
%                 end
                
                
                
                option_soc.F_manpg = F_manpg(test_random);
                option_soc.X_manpg = X_manpg;
                option_PAMAL.F_manpg = F_manpg(test_random);
                option_PAMAL.X_manpg = X_manpg;
                
                [X_Soc, F_soc(test_random),sparsity_soc(test_random),time_soc(test_random),...
                    soc_error_XPQ(test_random),maxit_att_soc(test_random),succ_flag_SOC]= soc_CM(H,option_soc);
                if succ_flag_SOC == 1
                    succ_no_SOC = succ_no_SOC + 1;
                end
                
                
                [X_pamal, F_pamal(test_random),sparsity_pamal(test_random),time_pamal(test_random),...
                    pam_error_XPQ(test_random), maxit_att_pamal(test_random),succ_flag_PAMAL]= PAMAL_CMs(H,option_PAMAL,V);
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
                
                
            end
            
        end
        
    end
    
    
    
    Result(index,1) = sum(lins)/succ_no_manpg;  Result(index,2) = sum(in_av)/succ_no_manpg;
    Result(index,3) = sum(lins_adap)/succ_no_manpg;  Result(index,4) = sum(in_av_adap)/succ_no_manpg;
    
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
    
    
    Sp.manpg(id_n) =  sum(sparsity_manpg)/succ_no_manpg;
    Sp.manpg_BB(id_n) =  sum(sparsity_manpg_BB)/succ_no_manpg_BB;
    Sp.soc(id_n) =  sum(sparsity_soc)/succ_no_SOC;
    Sp.pamal(id_n) =  sum(sparsity_pamal)/succ_no_PAMAL;
    Sp.Rsub(id_n) =  sum(sparsity_Rsub)/succ_no_sub;
    %fprintf(fid,' Alg ****  Iter *****  Fval *** sparsity ** cpu *** Error ***\n');
    
    %print_format =  'ManPG: %1.3e  %1.5e    %1.2f      %3.2f \n';
    %fprintf(fid,print_format, iter.manpg(id_n), Fval.manpg(id_n), Sp.manpg(id_n),time.manpg(id_n));
    %print_format =  'ManPG_adap: %1.3e  %1.5e    %1.2f      %3.2f \n';
    %fprintf(fid,print_format, iter.manpg_BB(id_n), Fval.manpg_BB(id_n), Sp.manpg_BB(id_n),time.manpg_BB(id_n));
    %print_format =  'SOC:   %1.3e  %1.5e    %1.2f      %3.2f    %1.3e\n';
    %fprintf(fid,print_format,iter.soc(id_n) , Fval.soc(id_n), Sp.soc(id_n) ,time.soc(id_n),mean(soc_error_XPQ));
    %print_format =  'PAMAL: %1.3e  %1.5e    %1.2f      %3.2f    %1.3e \n';
    %fprintf(fid,print_format,iter.pamal(id_n) ,  Fval.pamal(id_n) ,Sp.pamal(id_n),time.pamal(id_n),mean(pam_error_XPQ));
    %print_format =  'Rsub: %1.3e  %1.5e    %1.2f      %3.2f  \n';
    %fprintf(fid,print_format,iter.Rsub(id_n) ,  Fval.Rsub(id_n) ,Sp.Rsub(id_n),time.Rsub(id_n));
    %% plot
    fprintf(fid,' ---------------------------------------------------------\n');
    H =  - H;
    [VV,D] = eig(H+0);
    D = diag(D);
    [D I]= sort(D,'ascend');
    VV = VV(:,I);
    Density_True = sum(VV(:,1:r).*VV(:,1:r),2);
    fprintf('True Totoal Energy %2.5f \n',trace(VV(:,1:r)'*H*VV(:,1:r)));
    fprintf('Total Energy from CMs %2.5f \n',trace(X_manpg_adap'*H*X_manpg_adap));
    %flip the negative columns of CMs to be positive
    X_manpg = positive_CMs(X_manpg,r);
    X_manpg_adap = positive_CMs(X_manpg_adap,r);
    X_Soc = positive_CMs(X_Soc,r);
    X_pamal = positive_CMs(X_pamal,r);
    
    % CMs support - manpg-adap
    colormatrix = [0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 .1 0.5;.1 .1 .6;1,0.5 0;0.5 0.5 0.5];
    figure(1);
    set(gcf,'Position',[324   455   905   322]);
    set(gca, 'colororder',colormatrix, 'nextplot', 'replacechildren');
    x = [0:dx:L-dx]';
    plot(x,X_manpg_adap/sqrt(dx),'LineWidth',2);
    hlen = legend('\psi_1','\psi_2','\psi_3','\psi_4','\psi_5','Location','Bestoutside');
    set(hlen,'FontSize',20);
    title([' \mu = ' num2str(mu) ',  CMs  \phi'],'FontSize',25);
    set(gca,'FontSize',20);
    set(gcf, 'PaperPositionMode', 'auto');
    
    
    
    % CMs support - SOC
    colormatrix = [0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 .1 0.5;.1 .1 .6;1,0.5 0;0.5 0.5 0.5];
    figure(2);
    set(gcf,'Position',[324   455   905   322]);
    set(gca, 'colororder',colormatrix, 'nextplot', 'replacechildren');
    x = [0:dx:L-dx]';
    plot(x,X_Soc/sqrt(dx),'LineWidth',2);
    hlen = legend('\psi_1','\psi_2','\psi_3','\psi_4','\psi_5','Location','Bestoutside');
    set(hlen,'FontSize',20);
    title([' \mu = ' num2str(mu) ',  CMs  \phi'],'FontSize',25);
    set(gca,'FontSize',20);
    set(gcf, 'PaperPositionMode', 'auto');
    
    
    
    % CMs support - -PAMAL
    colormatrix = [0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 .1 0.5;.1 .1 .6;1,0.5 0;0.5 0.5 0.5];
    figure(3);
    set(gcf,'Position',[324   455   905   322]);
    set(gca, 'colororder',colormatrix, 'nextplot', 'replacechildren');
    x = [0:dx:L-dx]';
    plot(x,X_pamal/sqrt(dx),'LineWidth',2);
    hlen = legend('\psi_1','\psi_2','\psi_3','\psi_4','\psi_5','Location','Bestoutside');
    set(hlen,'FontSize',20);
    title([' \mu = ' num2str(mu) ',  CMs  \phi'],'FontSize',25);
    set(gca,'FontSize',20);
    set(gcf, 'PaperPositionMode', 'auto');
    
    
    % eigens of phi'*H*phi- manpg-adap
    
    [U1 D1] = eig(X_manpg_adap'*H*X_manpg_adap);
    D1 = diag(D1);
    [D1 ind1] = sort(D1,'ascend');
    U1 = U1(:,ind1);
    [U2 D2] = eig(X_Soc'*H*X_Soc);
    D2 = diag(D2);
    [D2 ind2] = sort(D2,'ascend');
    [U3 D3] = eig(X_pamal'*H*X_pamal);
    D3 = diag(D3);
    [D3 ind3] = sort(D3,'ascend');
    figure(4);
   % set(gcf,'Position',[324   455   905   322]);
    plot(D(1:r),'g*','LineWidth',1); hold on;
    plot(D1(1:r),'ko','LineWidth',1,'MarkerSize',10);
    hold on;
    plot(D2(1:r),'bd','LineWidth',1)
    hold on;
    plot(D3(1:r),'c+','LineWidth',1)
    
    set(gcf, 'PaperPositionMode', 'auto');
    hlen = legend('H','ManPG-Ada','SOC','PAMAL','Location','Best');
    set(hlen,'FontSize',20);
    set(gca,'FontSize',15);
    
    
end



