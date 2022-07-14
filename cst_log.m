function [j, grad] = cost_log(theta, X, y)
    m = length(y);

    %//cost calculation
    h = sigmoid(X*theta);
    t1 = -(1/m)*((y')*(log(h)));                             %'))
    t2 = -(1/m)*(((1-y)')*(log(1-h)));                       %'))
    j = t1+t2;

    %// gradient calculation
    grad = (1/m)*((X')*(h - y));                             %'))
    
end