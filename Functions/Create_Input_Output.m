function [INPUT,OUTPUT]= Create_Input_Output(data,IP)
%==========================================================================
% Creation of the matrices of input and output for the ANN
%==========================================================================
% Input: data--> matrix of data [N x M]
%        IP--> estimated optimal order of VAR model
% Output:INPUT--> matrix of regressors [M*IP, N-IP]
%        OUTPUT--> matrix of responses [M x N-IP]
%==========================================================================

p=IP;
Y=data;
[N,M]=size(Y); 
inputemp=[];

for j=1:N-p
    tempinput=Y(j:j+p-1,:);
    tempoutput=Y(p+j,:)';
    for o=p:-1:1
        inputemp=[inputemp;tempinput(o,:)'];
        
    end
    
    output1(:,j)=tempoutput;
    input1(:,j)=inputemp;
    inputemp=[];
end

INPUT=input1;
OUTPUT=output1;