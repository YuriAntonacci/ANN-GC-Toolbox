%% conditional GC ANALYSIS OF BRAIN-BODY INTERACTIONS IN REST/STRESS STATES

clear; close all; clc;

% Load data
load('TimeSeriesStress.mat')
p=[3 3 3]; % model order for each condition 
M=size(TimeSeriesStress,2);
N=300;
% Defines brain and body soruces
ii=[1 2 3]; % body drivers
kk=[4 5 6 7];% brain drivers
jj=3; % target process 
Tar={'eta','rho','pi','delta','theta','alpha','beta'};
Cond={'rest','ment','game'};
%% conditional GC computation - OLS -

for i_cond=1:3
    data_cond=TimeSeriesStress(:,:,i_cond);
    data_cond=zscore(data_cond,0,1);
    
    % MVAR model identification
    [Am_OLS,Su_OLS,Yp_OLS,Up_OLS,Z_OLS,Yb_OLS]=idMVAR(data_cond(1:N,:)',p(i_cond),0);
    
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
                Fi_js(jj,ii)=log(Sj_js/Sj_ijs);
                
            end
        end
    end
    
    cGC=Fi_js;
    [Fi_jsSurr]=cGCsurrogate(data_cond,100,p(i_cond));
    thr=prctile(Fi_jsSurr,95,3);
    cGC(cGC<=thr)=0;
    OLS(:,:,i_cond)=cGC;
    
end

%% conditional GC computation - ANN -

%%% ANN paramters
lambda=logspace(1,3,300);
train_p=90;
Ntrain=2000;
lr=10^-3;
crit=1; % knee of the GCV curve
for i_cond=1:3
    data_cond=TimeSeriesStress(:,:,i_cond);
    data_cond=zscore(data_cond,0,1);
    %%% create input and output matrices
    [INPUT,OUTPUT]= Create_Input_Output(data_cond,p(i_cond));
    %%% create input and output for training and testing sets
    [IN_train,OUT_train,IN_test,OUT_test]=Create_train_test_sets(INPUT,OUTPUT,train_p,1);
    
    %%% initialization of the weights parameters
    weights_init=zeros(M*p(i_cond)+1,size(OUTPUT,1));
    %%% training of the neural network
    [VARopt,ind,RSS,GCV,df]= GCV_ANN(IN_train,OUT_train,IN_test,OUT_test,weights_init,Ntrain,lambda,lr,crit);
    lopt=lambda(ind);
    %%% residual covariance matrix
    Am=reshape(VARopt,M,[]);
    [S]=cov_residual(data_cond',Am);
    
    
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
                Fi_js(jj,ii)=log(Sj_js/Sj_ijs);
                
            end
        end
    end
    
    
    ANN(:,:,i_cond)=Fi_js;
    
    
end

%% plot estimated cGC networks
for i_cond=1:3
    figure
    subplot(1,2,1);
    plot_pw(OLS(:,:,i_cond));
    title('OLS');
    subplot(1,2,2);
    plot_pw(ANN(:,:,i_cond));
    title('ANN')
    tit=sprintf('Cond: %s',Cond{i_cond});
    suptitle(tit)
end