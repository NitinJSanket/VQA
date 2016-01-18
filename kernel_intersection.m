function K = kernel_intersection(X, X2)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
% K = KERNEL_INTERSECTION(X, X2)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the histogram
% intersection kernel.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% X =X';
% X2 =X2';

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.

% for i = 1:m
%     for j = 1:n
%         a = min(X2(:,i),X(:,j));
%         K(i,j) = sum(a);
%     end
% end

for i = 1:n
Xi = X(i,:);
Xirep = repmat(Xi,[m,1]);
Ktemp = sum(bsxfun(@min,X2,Xirep),2);
K(:,i) = Ktemp;
end