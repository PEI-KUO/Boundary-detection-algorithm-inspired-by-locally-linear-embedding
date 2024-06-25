function [b,c] = LLE_indicator_k_regularizer(data,k,d)

% INPUT
% d: dimension of the manifold
% OUTPUT
% b:bump function; c:regularizer
[N,D] = size(data); % N: sample size; D: data in R^D
[index,sorted]= knnsearch(data, data, 'k', floor(N/10))


epsilon=quantile(sorted(:,k+1),0.5)

val=zeros(d+2,N); %record largest d+2 eigenvalues to chose the regularizer
for ii=1:N
   j = index(ii,2:k+1);
   z = data(j,:)-repmat(data(ii,:),k,1); % local data matrix (shift ith pt to origin)
   C=z*z';
   val(:,ii)=eigs(C,d+2)
end;      

mean_val=mean(val,2);
c=sqrt(mean_val(d+1)*mean_val((d))) % choose the regularizer to be geometric mean

for ii=1:N
   j = index(ii,2:k+1);
   z = data(j,:)-repmat(data(ii,:),k,1); % local data matrix (shift ith pt to origin)
   C=z*z';
   C = C + c*eye(k,k);
   w = C\ones(k,1);         % solve Cw=1
   b(ii)=1- c*sum(w)/k;     % indicator function
end;      
end