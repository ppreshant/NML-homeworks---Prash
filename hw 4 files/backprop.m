function M=backprop(X,Y)
	% simple network - 1 hidden layer 2 PE ; 2 inputs ; 1 output (include 2 bias units in input and hidden layer)
    mu = .001;n = 100;tol = .1;  % mu = learning rate, n - max iterations for convergence, tol = tolerance
    [nx,mx]=size(X);  % x = input  - 3 x 1
	[ny,my]=size(Y);  % y = output   1 x 1
    
%initialize the memory matrices
    W1 = -1 + 2* rand(2,3) ; W2 = -1 + 2* rand(1,3);  % weights of layer 1 and layer 2
%     M = {W1,W2}; % weights of 
    M=zeros(ny,nx);
	for k=1:n %n is the maximum number of outer loop iterations
% randomize the ordering of the vector in both X & Y
        RN=randperm(mx); X=X(:, RN); Y=Y(:, RN); % randomize patterns
        for i = 1:p % forward run
            good morning
        end
        for i = p:1 % backprop
            good night
        end
        if norm(Y-M*X)<=tol
			disp('Gradient Search Terminated ===>>> ||Y-M*X||<=tol')
			disp('Number of Iterations = '), disp(n*i)
			break
        end
    end
    
        