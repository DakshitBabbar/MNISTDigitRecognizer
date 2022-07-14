%//Recognizing Handwritten Digits
%//data used for this project is a subset of the MNIST handwritten digit dataset at http://yann.lecun.com/exdb/mnist/
%//user should work in the directory where the data is present
clear;
close all;

%%=============================================================%%
%//preparing common data for all the models described below
fprintf('Preparing data:\n');

%//.mat files automatically load data into the octave environment
%//here it loads the data into
%//X: a 5000 x 400 matrix with each row as an unrolled form of one training example having 20 x 20 pixel image of a digit
%//y: a 5000 x 1 vector with each element from the set {1, 2, ... , 10} representing the lables of the examples (10 -> digit zero)
%//we divide X into two data sets X(4500 x 400) and X_test(500 x 400) and similarly with y.
%//We use X as our training data and X_test as our testing or validation data
load('Data.mat');
r = randperm(5000);
X_test = X(r(1:500), :);
y_test = y(r(1:500), 1);
X = X(r(501:end), :);
y = y(r(501:end), 1);
[m,n] = size(X);
k = 10;
fprintf('Data loaded, press enter to continue\n\n');
pause;

fprintf('Press enter to visualize data\n');
pause;

rand_ind = randperm(m);
vis_data(X(rand_ind(1:100), :));
fprintf('Data prepared, press enter to continue\n\n');
pause;

%%=============================================================%%
%//using Multiclass Logistic Regression without regularization
fprintf('=========== Multiclass Logistic Regrssion without regularization =========\n');

%//model design
all_theta = zeros(k, n+1);

fprintf('Training model...\n');
%//training
all_theta = train_log(X,y,all_theta);
fprintf('Training complete, press enter to continue\n');
pause;


fprintf('Predicting values and calculating accuracy...\n');
%//predicting values
pred = pred_log(X,all_theta);
pred_test = pred_log(X_test, all_theta);

acc = (mean(double(pred==y)))*100;
acc_test = (mean(double(pred_test==y_test)))*100;
fprintf('Accuracy on the training data = %f %%\n', acc);
fprintf('Accuracy on the test data = %f %%\n', acc_test);
fprintf('For next model, press enter\n\n');
pause;

%%=============================================================%%
%//using Multiclass Logistic Regression with regularization
fprintf('============ Multiclass Logistic Regrssion with regularization ==========\n');

%//model design
all_theta = zeros(k, n+1);

fprintf('Training model...\n');
%//training
all_theta = train_log_reg(X,y,all_theta);
fprintf('Training complete, press enter to continue\n');
pause;


fprintf('Predicting values and calculating accuracy...\n');
%//predicting values
pred = pred_log_reg(X,all_theta);
pred_test = pred_log_reg(X_test, all_theta);

acc = (mean(double(pred==y)))*100;
acc_test = (mean(double(pred_test==y_test)))*100;
fprintf('Accuracy on the training data = %f %%\n', acc);
fprintf('Accuracy on the test data = %f %%\n', acc_test);
fprintf('For next model, press enter\n\n');
pause;

%%=============================================================%%
%//using artificial neural networks
fprintf('====================== Artificial Neural Networks ======================\n');




