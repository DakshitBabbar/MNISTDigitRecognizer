function [j, grad] = cost_log_reg(theta, X, y, lambda)
    m = length(y);

    %//cost calculation
    h = sigmoid(X*theta);
    t1 = -(1/m)*((y')*(log(h)));                             %'))
    t2 = -(1/m)*(((1-y)')*(log(1-h)));                       %'))

    v = (lambda/(2*m))*((theta(2:end, 1)).^2);
    t3 = sum(v);

    j = t1+t2+t3;

    %// gradient calculation
    grad = (1/m)*((X')*(h - y) + lambda*theta);                             %'))
    grad(1,1) = grad(1,1) - (lambda/m)*(theta(1,1));
    
end