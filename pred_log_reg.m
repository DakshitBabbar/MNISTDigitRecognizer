function pred = pred_log_reg(X, all_theta)
    X = [ones(size(X,1),1) X];

    all_pred = X*(all_theta');              %')
    [m idx] = max(all_pred, [], 2);
    pred = idx;
end