function M=backprop(X,Y)
	% simple network - 1 hidden layer 2 PE ; 2 inputs ; 1 output (include 2 bias units in input and hidden layer)
    mu = .001;n = 100;tol = .1;  % mu = learning rate, n - max iterations for convergence, tol = tolerance
    [nx,mx]=size(X);  % x = input  - 3 x 4 (4 different patterns)
	[ny,my]=size(Y);  % y = output   1 x 4 (4 different patterns)
    
%initialize the memory matrices
    W1 = -1 + 2* rand(2,3) ; W2 = -1 + 2* rand(1,3);  % weights of layer 1 and layer 2
%     M = {W1,W2}; % weights of 
    M=zeros(ny,nx);
	for ou=1:n %n is the maximum number of outer loop iterations
% randomize the ordering of the vector in both X & Y
        RN=randperm(mx); X=X(:, RN); Y=Y(:, RN); % randomize patterns
         % run through all patterns now in random order
        for k = 1:mx 
            % forward run
            x = X(:,k); y = Y(:,k); % x is 3x1 and y is 1x1
            y1 = tanh(W1*x); y1(3) = 1 ; y2 = tanh(W2*y1);     % calculate the outputs y1=2x1 -> 3x1 w bias, y = 1x1
                           % add bias value y1 = 3 x 1
            % backprop
            e2 = (y - y2); delta2 = e2*(1 - y2^2);  % calculating e2 and Delta2 -> for tanhx: f' = 1 - f^2  
            delta1 = (1-y1^2)*delta2*W2(1:2,1);     % the backpropegation step  
            W2 = W2 + mu * delta2 * y1;
            W1 = W1 + mu * delta1 * x;
            
        end
        for i = p:1 
            good night
        end
        if norm(Y-M*X)<=tol
			disp('Gradient Search Terminated ===>>> ||Y-M*X||<=tol')
			disp('Number of Iterations = '), disp(ou)
			break
        end
    end
    
        