function vis_data(X)
    [m,n] = size(X);
    ex_width = round(sqrt(n));
    ex_height = n/ex_width;

    grid_width = round(sqrt(m));
    grid_height = round(m/grid_width);

    colormap(ocean);
    pad = 1;

    rows = pad + grid_height*(ex_height + pad);
    cols = pad + grid_width*(ex_width + pad);
    grid_array = -ones(rows, cols);

    ex = 1;
    for i=1:grid_height;
        for j=1:grid_width;
            if(ex>m),
                break;
            end
            r_add = pad + (i-1)*(ex_height + pad);
            c_add = pad + (j-1)*(ex_width + pad);

            rs = (1:ex_height) + r_add;
            cs = (1:ex_width) + c_add;

            div = max(abs(X(ex,:)));
            fill = reshape(X(ex,:), ex_height, ex_width)/div;
            grid_array(rs,cs) = fill;

            ex = ex + 1;
        end

        if(ex>m)
            break;
        end
    end

    imagesc(grid_array, [-1,1]);
    axis image off;
    drawnow;

end