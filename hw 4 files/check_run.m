function ou = check_run(W1,W1,X,Y)
    [nx,mx]=size(X);  % x = input  - 3 x 4 (4 different patterns)
	% y = output   1 x 4 (4 different patterns)
    err = zeros(1,mx);
    for i= 1:mx
        outpt = tanh(W2*[tanh(W1*X(:,i));1]);
        desired_output = Y(i);
        err(i) = desired_output - outpt;  
    end
end