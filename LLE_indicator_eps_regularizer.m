function [b,c] = LLE_indicator_eps_regularizer(data,epsilon,d)

% INPUT
% d: dimension of the manifold
% OUTPUT
% b:bump function; c:regularizer
[N,D] = size(data); % N: sample size; D: data in R^D
[index,sorted]=rangesearch(data,data,epsilon); % find epsilon balls


val=zeros(d+1,N); %record largest d+2 eigenvalues to chose the regularizer
for ii=1:N
    j = index{ii};
    K = length(j) ;
    z = data(j',:)-repmat(data(ii,:),K,1); % local data matrix (shift ith pt to origin)
    C=z*z';
    val(:,ii)=eigs(C,d+1)
end; 
mean_val=mean(val,2);
c=sqrt(mean_val(d+1)*mean_val((d))) % choose the regularizer to be geometric mean

for ii=1:N
    j = index{ii};
    K = length(j) ;
    z = data(j',:)-repmat(data(ii,:),K,1); % local data matrix (shift ith pt to origin)
    C=z*z';
    C = C + c*eye(K,K);
    w = C\ones(K,1);         % solve Cw=1
    b(ii)=1- c*sum(w)/K;     % indicator function
end;      
end