%%% Test_Oscillators - Analysis of unconditional GC on a ring of 32 chaotic
%%% oscillators - Synchronizaion analalysis.

%%% load data
load('top_row.mat');

data=double(ts);
fs=100000; % 100 Khz sampling frequency
w=8000; % window to plot
%%% decimation of time series
for tt=1:size(data,1)
    data1(tt,:) = decimate(data(tt,:)',4,'fir');
end
%%% second step of decimation
for tt=1:size(data,1)
    data2(tt,:) = decimate(data1(tt,:)',10,'fir');
end
%%% Cross-correlation analysis

for gg=1:size(ts,1)
    % Hilbert transform
    hx = hilbert(data(gg,:)');
    % calculating the INSTANTANEOUS AMPLITUDE (ENVELOPE)
    inst_amp(:,gg) = abs(hx);
    
end

%%% cross correlation between instantaneous amplitudes
for uu=1:size(ts,1)
    for yy=1:size(ts,1)
        [XX,LAG] = xcorr(inst_amp(:,uu)',inst_amp(:,yy)','normalized');
        [TAU]=find(LAG>=0);
        C(uu,yy)=max(abs(XX(TAU(1):end)));
        [ind(uu,yy)]=find(abs(XX(TAU(1):end))==max(abs(XX(TAU(1):end))));
        
    end
    Cij(uu,:)=mat2gray(C(uu,:)); %normalization 
end
%%% plot of the cross correlation
figure
imagesc(Cij)
colormap(jet)
colorbar
set(gca,'YTick',[1:32],'YTickLabel',{'1','','','','','','','','','','','','','','','16','','','','','','','','','','','','','','','','32'});
set(gca,'XTick',[1:32],'XTickLabel',{'1','','','','','','','','','','','','','','','16','','','','','','','','','','','','','','','','32'});
xlabel('Node i','FontSize',14,'FontName','TimesNewRoman')
ylabel('Node j','FontSize',14,'FontName','TimesNewRoman')

%%% plot of the instanteneous amplitude synchronization
inst_amp=inst_amp';
node_i=1;
node_j=[2 6 9];
figure
for oo=1:length(node_j)
%%% realignment respct to the positive lag for which we have max(Cij)
[X,Y] = alignsignals(inst_amp(node_i,ind(node_j(oo),1):w+ind(node_j(oo),1)),inst_amp(node_j(oo),1:w+ind(node_j(oo),1)),ind(node_j(oo),1));
subplot(3,1,oo)
plot(zscore(X',0,1),'b');hold on;plot(zscore(Y',0,1),'r');
xlim([ind(node_j(oo),1) w])
end

%%% Granger Causality Analysis
%%% !!! IT TAKES A LONG TIME THE SS REPRESENTATION

popt=16; % verified with the previous line
M=size(data2,1);
Y=zscore(data2,0,2)';
tic
%%% VAR model Identification
[Am,S]=idMVAR(Y',popt,0); %OLS 
%%% State space representation of VAR model
[A,C,K,V,Vy] = varma2iss(Am,[],S,eye(M));
% Unconditional GC 
for jj=1:M
    disp(jj)
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii]);
            Sj_ijs=tmp(1,1);
            Fi_j(jj,ii)=log(Sj_js/Sj_ijs);
            
        end
    end
end
OLS=Fi_j;
TIME_OLS=toc;

figure
imagesc(OLS)
colormap(jet)
set(gca,'YTick',[1:32],'YTickLabel',{'1','','','','','','','','','','','','','','','16','','','','','','','','','','','','','','','','32'});
set(gca,'XTick',[1:32],'XTickLabel',{'1','','','','','','','','','','','','','','','16','','','','','','','','','','','','','','','','32'});
xlabel('Driver (i)','FontSize',14,'FontName','TimesnewRoman')
ylabel('Target (j)','FontSize',14,'FontName','TimesnewRoman')

%%% Unconditional GC using ANNs
lambda=logspace(1,3,300);
train_p=90;

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

%%% Unconditional GC
for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Fi_j(jj,ii)=log(Sj_js/Sj_ijs);
            
        end
    end
end
TIME_NN=toc;
ANN=Fi_j;
figure
imagesc(ANN)
colormap(jet)
set(gca,'YTick',[1:32],'YTickLabel',{'1','','','','','','','','','','','','','','','16','','','','','','','','','','','','','','','','32'});
set(gca,'XTick',[1:32],'XTickLabel',{'1','','','','','','','','','','','','','','','16','','','','','','','','','','','','','','','','32'});
xlabel('Driver (i)','FontSize',14,'FontName','TimesnewRoman')
ylabel('Target (j)','FontSize',14,'FontName','TimesnewRoman')
