%% make XOR patterns
% X = [0 0 1;1 0 1;0 1 1;1 1 1]';
% Y = [0 1 1 0];
X = [-1 -1 1;1 -1 1;-1 1 1;1 1 1]';
Y = [-1 1 1 -1];
%% Run backprog algo
[w1,w2,Erms] = bare_backprop(X,Y);
%% Verify result
outpt = tanh(w2*[tanh(w1*X);1 1 1 1])
desired_output = Y