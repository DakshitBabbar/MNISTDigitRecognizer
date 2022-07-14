function ans = train_log_reg(X,y,all_theta)
    ans = zeros(size(all_theta));
    k = size(all_theta, 1);

    X = [ones(size(X,1),1) X];

    lambda = 3;

    for cls = 1:k
        %//set data for class cls
        yc = (y==cls);

        %//set model for class cls
        theta_c = (all_theta(cls,:))';          %'

        %//train
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        [temp cost] = fminunc(@(t)(cst_log_reg(t,X,yc,lambda)), theta_c, options);

        ans(cls,:) = temp';                     %'
    end
end