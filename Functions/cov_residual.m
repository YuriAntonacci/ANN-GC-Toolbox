function [S]=cov_residual(Y,Am)
% Estimation of the residual covariance matrix
%==========================================================================
% Input: Y--> matrix of data [M x N]
%        Am--> estimated optimal order of VAR model
% Output:S--> residual covariance matrix [M x M]
%==========================================================================

M=size(Y,1);
p=size(Am,2)/M;
N=size(Y,2);

Z=NaN*ones(p*M,N-p); % observation matrix
for j=1:p
    for i=1:M
        Z((j-1)*M+i,1:N-p)=Y(i, p+1-j:N-j);
    end
end
Yp=Am*Z; 
Yp=[NaN*ones(M,p) Yp]; % Vector of predicted data
Up=Y-Yp; Up=Up(:,p+1:N); % residuals of strictly causal model
S=cov(Up');
