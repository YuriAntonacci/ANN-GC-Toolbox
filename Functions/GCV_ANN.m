function [VARopt,ind,RSS,GCV,df]= GCV_ANN(IN_train,OUT_train,IN_test,OUT_test,weights_init,Ntrain,lambda,lr,crit)
% training and selection of the regularization paramter
%==========================================================================
%Input:  - IN_train--> matrix of Input samples for traininng [M*IP x N]
%        - OUT_train--> matrix of Input samples for traininng [M x N]
%        - IN_test--> matrix of the inputs of the test set
%        - OUT_test--> matrix of the ouputs of the test set
%        - weights_init--> initialzed weights of the networks
%        - Ntrain--> Number of iterations for the SGD-L1
%        - lambda--> vector of the all possible values of lambda to be tested
%        - lr--> learning rate
%        - crit--> criterion for the minimum (0 - for GCV, 1 - knee of the GCV curve)
%Output: 
%        - VARopt--> matrix of the VAR coefficients for the optimal lambda
%        - ind--> index of the optimal lambda value
%        - RSS--> residual sum of square
%        - GCV--> GCV function per ogni lambda
%        - df--> vector of degrees of freedom for each lambda
%==========================================================================

for ll=1:length(lambda)
    
    if ll==1
        [WW] = traininig_SGD_L1(IN_train',OUT_train',weights_init,Ntrain,lambda(ll),lr);
    else
        [WW] = traininig_SGD_L1(IN_train',OUT_train',WW,Ntrain,lambda(ll),lr);
    end
    WW1=WW(1:size(IN_train,1),:); %parameters on the training set
    MVARp1=reshape(WW1',size(OUT_train,1),size(OUT_train,1),[]);
    MVARp(:,:,:,ll)=MVARp1;
        
    b_NN=WW1;
    DF=nnz(b_NN);
    df(ll)=DF;
    [OUTPUT_P] = Predict_output(IN_test',WW);
    RSS(ll) = mean(sum(abs(OUT_test'-OUTPUT_P).^2, 1));

end

NN=numel(b_NN); %numero massimo di non zeri possibili
GCV= RSS./(NN-df).^2;
if crit==1
[~, ind] = knee_pt(log10(GCV));
else
[~, ind] = min(GCV);
end
VARopt=MVARp(:,:,:,ind);
end

        