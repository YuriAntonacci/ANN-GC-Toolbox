%% Conditional Granger Causality - theoretical example -
% analysis of simulated 5-variate VAR process

clear; close all; clc;

load('TimeSeries.mat')

%%% MVAR process parameters

M=size(Am,1);
Su=eye(M);
p=size(Am,2)/M;
N=300; % Number of data samples (set the desired)
kratio=(N*M)/(M*M*p);
Y1=Y;
Y1=zscore(Y1(1:N*1.11,:),0,1);
Y=zscore(Y(1:N,:),0,1);
% Y1=Y1(1:N*1.11,:);
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
lambda=logspace(1,3,300); % interval of lambdas

%%% create input and output matrices
[INPUT,OUTPUT]= Create_Input_Output(Y,p);
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
            Fi_js(jj,ii)=log(round(Sj_js,15)/round(Sj_ijs,15));

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