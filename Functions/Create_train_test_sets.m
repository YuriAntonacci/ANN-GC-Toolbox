function [IN_train,OUT_train,IN_test,OUT_test]=Create_train_test_sets(INPUT,OUTPUT,train_p,stand)
%==========================================================================
%Input: - INPUT--> matrix of INPUT (lagged variables) [M*IP x N]
%       - OUTPUT--> matrix of OUTPUT (responses variables) [M x N]
%       - train_p--> percentage of the observations for the training (ex 0.9)
%       - stand---> standardization for the input/output matrices ('1' yes)
%Output:
%        - IN-train--> Matrix of inputs for the training
%        - OUT_train--> Matrix of outputs for the training
%        - IN_test--> Matrix of inputs for the testing procedure
%        - OUT_test--> Matrix of outputs for the testing procedure
%==========================================================================

ind=randperm(size(OUTPUT,2));
train_p=round(length(ind)*(train_p/100));
IN_train=INPUT(:,ind(1:train_p));
OUT_train=OUTPUT(:,ind(1:train_p));
IN_test=INPUT(:,ind(train_p+1:end));
OUT_test=OUTPUT(:,ind(train_p+1:end));

if stand==1
    IN_train=zscore(IN_train,0,2);
    OUT_train=zscore(OUT_train,0,2);
    IN_test=zscore(IN_test,0,2);
    OUT_test=zscore(OUT_test,0,2);
else
    
    
end
