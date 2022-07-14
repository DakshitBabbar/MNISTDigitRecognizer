function g = sigmoid(z)
    pwr = exp(-z);
    deno = 1+pwr;
    g = 1./deno;
end