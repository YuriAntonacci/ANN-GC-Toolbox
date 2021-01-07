%% Conditional Granger Causality - theoretical example -
% analysis of simulated 10-variate VAR process as described in the main
% document - Simulation study 3 -

clear; close all; clc;
%%% Time series generation
%%% part of code ispired from Pascucci et al.,2020 for stationary time
%%% series

M=10; % number of processes
Fs=200; % sampling frequency
popt=6; % number of lags
K=100;  % desidered ratio between data samples and AR coefficients
Nsamp=K*(M*popt);
dur=Nsamp/Fs; % duration in s
sparsity=0.5;  % percentage of connections different from 0 
% - constants
SClinks    = 0.8;                         % Markov et al., 2012
ascale     = 0.15:0.01:0.5;
dt         = 1/Fs;
time       = 0:dt:(dur-dt);
samples    = numel(time);
p          = popt;

% - structural links (binary mask)
I          = eye(M);
UT         = triu(reshape(randsample([0 1],M^2,'true',...
    [1-SClinks SClinks]),[M M]));
MK         = (UT+UT') - diag(diag(UT+UT'));
SC         = MK + I;

% - directed interactions (binary mask)
FC         = zeros(size(SC));
FC(randsample(find(MK),fix((1-sparsity)*numel(find(MK))),'false')) = 1;
FC         = FC + I;

% - AR process (univariate)
AR         = zeros(M,M,popt);
for i = 1:M
    c1            = randsample(ascale,1);
    c2            = randsample(ascale,1);
    AR(i,i,[1]) = [c1]; % low-pass
end

% - AR process (interactions)
cON        = find(MK.*FC);
scalef           = 1.5;
summary_conn     = table();
ok           = 0;
summary      = NaN(numel(cON),6);
while ok==0
    % generate off-diag AR and check stability
    tmpAR      = AR;
    for ij = 1:numel(cON)
        ij_p   = randsample(1:p-1,1);
        ampl1  = randsample(ascale,1);                              % scaling off diag?
        ampl2  = randsample(ascale,1);
        osc    = sign(randn(1,2)).*[ampl1 ampl2];%.*scalef;
        [i,j]  = ind2sub([M M],cON(ij));
        summary(ij,:) = [i j ampl1 osc ij_p];
        tmpAR(i,j,ij_p:ij_p+1)  = osc;
    end
    pp         = p;
    % stability check
    blockA     = [tmpAR(:,:); eye((pp-1)*M) zeros((pp-1)*M,M)];
    if any(abs(eig(blockA))>.95)
        scalef = scalef*.95;
        clear tmpAR blockA
    else
        ok     = 1;
    end
end
head= {'rec','send','mag','osc','lagop'};
tmp= table(summary(:,1),summary(:,2),summary(:,3),summary(:,4:5),summary(:,6),'VariableNames',head);
summary_conn   = vertcat(summary_conn,tmp);
AR= tmpAR;
% Residuals covariance matrix.
SIGT = eye(M);
% Generate VAR time series data with normally distributed residuals
% for specified coefficients and covariance matrix.
X = var_to_tsdata(AR,SIGT,Nsamp,1);
X=X';
%%% MVAR process parameters
Am=reshape(AR,M,[]);
Su=SIGT;
N=200; % Number of data samples (set the desired)
kratio=(N*M)/(M*M*p);
Y1=X;
Y1=zscore(Y1(1:N*1.11,:),0,1);
Y=zscore(X(1:N,:),0,1);

%% Theoretical conditional GC network

%%% ISS paramters
[A,C,K,V,Vy] = varma2iss(Am,[],Su,eye(M));

% % Conditional Granger Causality (eq. 12)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,15)/round(Sj_ijs,15));
            
        end
    end
end
THEO=Fi_js;

%% conditional GC network - OLS -

% MVAR model identification
[Am_OLS,Su_OLS,Yp_OLS,Up_OLS,Z_OLS,Yb_OLS]=idMVAR(Y',p,0);

%%% ISS paramters
[A,C,K,V,Vy] = varma2iss(Am_OLS,[],Su_OLS,eye(M));

% % Conditional Granger Causality (eq. 12)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,15)/round(Sj_ijs,15));
            
        end
    end
end
cGC=Fi_js;

% testing significance for the estimated cGC values with surrogates
[Fi_jsSurr]=cGCsurrogate(Y,100,p);
thr=prctile(Fi_jsSurr,95,3);
cGC(cGC<=thr)=0;
OLS=cGC;

%% conditional GC network - ANN -

%%% regularization paramter
lambda=logspace(0.5,2,300); % interval of lambdas

%%% create input and output matrices
[INPUT,OUTPUT]= Create_Input_Output(Y1,p);
%%% create input and output for training and testing sets
[IN_train,OUT_train,IN_test,OUT_test]=Create_train_test_sets(INPUT,OUTPUT,90,1);

%%% initialization of the weights parameters
weights_init=zeros(M*p+1,size(OUTPUT,1));

%%% training of the neural network
Ntrain=1000;
lr=10^-3;
crit=0; % GCV curve minimum
[VARopt,ind,RSS,GCV,df]= GCV_ANN(IN_train,OUT_train,IN_test,OUT_test,weights_init,Ntrain,lambda,lr,crit);
lopt=lambda(ind);

%%% residual covariance matrix
Am=reshape(VARopt,M,[]);
[S]=cov_residual(Y',Am);

%%% plot of GCV function
Fig1=figure;
set(Fig1(1),'Position',[196   469   560   418]);
plot(log10(lambda),GCV,'LineWidth',1.3)
xlabel('log( {\lambda} )');
ylabel('log (GCV)')
hold on
[ind]=find(lambda==lopt);
plot(log10(lopt),GCV(ind),'or','LineWidth',1.7)
tit=sprintf('Selected lambda = %s',num2str(lopt));
title(tit);

% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am,[],S,eye(M)); %


% % Conditional Granger Causality (eq. 12)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_js(jj,ii)=log(round(Sj_js,4)/round(Sj_ijs,4));
            
        end
    end
end
ANN=Fi_js;

%% plot of cGC networks

figure
subplot(1,3,1);
plot_pw(THEO);
title('Theo');
subplot(1,3,2);
plot_pw(OLS);
title('OLS');
subplot(1,3,3);
plot_pw(ANN);
title('ANN')
tit=sprintf('Samples=%s, Kratio=%s',num2str(N),num2str(kratio));
suptitle(tit)