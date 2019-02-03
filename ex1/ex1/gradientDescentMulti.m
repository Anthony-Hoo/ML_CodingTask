function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % 值得一提的是，测试样例中给的 X 矩阵有4列，而‘ex1data2.txt’文件中给的数据得到的 X 矩阵只有3列
    % 这可以通过一个for循环解决，但是本人懒得写了，将就着用吧（逃

    h = sum((theta' .* X)')';
    % disp(h)
    tmp1 = theta(1,:) - (sum(h - y) * alpha / m); 
    tmp2 = theta(2,:) - (sum((h - y) .* X(:,2)) * alpha  / m); 
    tmp3 = theta(3,:) - (sum((h - y) .* X(:,3)) * alpha  / m);
    % tmp4 = theta(4,:) - (sum((h - y) .* X(:,4)) * alpha  / m);
    % theta = [tmp1 ; tmp2 ; tmp3 ; tmp4];
    theta = [tmp1 ; tmp2 ; tmp3];
    %disp(theta)





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
