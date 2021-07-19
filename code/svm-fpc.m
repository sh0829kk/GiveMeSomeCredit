
close all;clear all; clc;
%warning off all;
function [beta,trainerr,testerr,traintime,testtime]=FPC(xtr,ytr,xte,yte,s,cx)
%%% Input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% xtr         -- inputs of training samples
%%% ytr         -- labels of training samples
%%% xte         -- inputs of testing samples
%%% yte         -- labels of testing samples
%%% s           -- algorithmic parameter: the order of the used polynomial (generally, an integer less than 10)
%%% cx          -- algorithmic parameter: center points in X-space (generally, taking the center points as the first nc columns of the polynomial kernel matrix)
%%% gamma       -- augmented lagrangian parameter (default:1)
%%% alpha       -- proximal parameter (defualt: 1), mainly to overcome the ill-condedness of the induced kernel matrix and improve the numerical stability
%%% MaxIter     -- the default number of maximal iterations: 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Output %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% beta        -- the parameters of model
%%% testerr     -- test error (misclassification rate)
%%% trainerr    -- training error (misclassification rate)
%%% testtime    -- test time (seconds)
%%% traintime   -- training time (seconds)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



start_cpu_time = cputime;
% the default parameters for FPC
alpha = 1; % proximal parameter
gamma = 1; % the augmented parameter
MaxIter = 50; % the maximal iterations

ATrain = PolynomialKerMat(xtr,cx,s); % the associated matrix via polynomial kernel using training samples
ATest = PolynomialKerMat(xte,cx,s); % the associated matrix via polynomial kernel using test samples
[m,n]=size(ATrain);

% calculate the inverse and restore
tempA = (gamma*(ATrain'*ATrain)+alpha*eye(n))\eye(n);

% initialization
u0=zeros(n,1);
v0=ytr;
w0=zeros(m,1);

iter = 1;
while iter<=MaxIter
    ut = tempA*(alpha*u0+ATrain'*(gamma*v0-w0)); % update ut
    vt = hinge_prox(ytr,ATrain*ut+gamma^(-1)*w0,m*gamma); % update vt
    wt = w0+gamma*(ATrain*ut-vt); % update multiplier wt
    
    u0 = ut;
    v0 = vt;
    w0 = wt;
    iter = iter +1;
end
end_cpu_time = cputime;
traintime = end_cpu_time - start_cpu_time; % calculating the training time

beta = u0;
trainerr = sum((ytr~=mysign(ATrain*u0)))/size(ytr,1); % calculating test error

start_test_time = cputime;
testerr = sum((yte~=mysign(ATest*u0)))/size(yte,1); % calculating test error
end_test_time = cputime;
testtime = end_test_time - start_test_time;
end


%%% Constructing a matrix given the center points of kernel x, coefficients c and the order of polynomial kernel s.
function A = PolynomialKerMat(x,c,s)
% c: Polynomial kernel coefficient
% s: the order of Polynomial kernel
A = (ones(size(x,1),size(c,1))+x*c').^s;
end

function tempsign = mysign(u)
tempsign = (u>=0)-(u<0);
end

function z = hinge_prox(a,b,gamma)
% hinge_prox(a,b,gamma) = argmin_u max{0,1-a*u} + gamma/2 (u-b)^2
% a: m*1 vector
% b: m*1 vector
% gamma>0: parameter
% z: output of the proximal of hinge
% m = size(a,1);
tol = 1e-10;
z = b.*(a==0)+(b+gamma^(-1)*a).*(a~=0&a.*b<=1-gamma^(-1)*a.*a)+...
    (a~=0&a.*b>1-gamma^(-1)*a.*a&a.*b<1).*((a+tol).^(-1))+b.*(a~=0&a.*b>=1);
end

%% Load dataset and data processing
%load('testpro2.mat');
% normalization: input [0,1], output {-1,1}
dataset = csvread('trainpro2.csv'); % file name of the dataset
% total number of samples 6598*167
% 2-class classification
% the last column is the target
% 50% for training, 25% for validation, 25% for test
% TrainDataSize =  3299;
% ValidDataSize = 1649
% TestDataSize = 1650;

% no. of negative labels: 5581
% no. of poisitve labels: 1017

%% data normalization:
%%%% input [0,1]
for i=1:size(dataset,2)-1
    if max(dataset(:,i))~=min(dataset(:,i))
      
        dataset(:,i)=((dataset(:,i)-min(dataset(:,i)))/(max(dataset(:,i))-min(dataset(:,i))));
    else
        dataset(:,i)=0;
    end
end

% output {-1,1}
T=dataset(:,size(dataset,2));
if max(T)~=min(T)
    T=2*(T-min(T))/(max(T)-min(T))-1;
else
    T=ones(size(T))*0.5;
end
dataset(:,size(dataset,2))=T;

%% Generate the training and test samples
[DataSize,d] = size(dataset);
m0 = 95000; % number of training samples
m1 = 37500; % number of validation samples
m2 = DataSize-m0-m1; % Number of Test samples

trail = 2; % 20 trails
trainerr = zeros(trail,1); % recording training error
testerr = zeros(trail,1); % recording test error
traintime = zeros(trail,1); % recording training time
testtime = zeros(trail,1); % recording test time
best_s = zeros(trail,1); % recording best polynomial degree s
best_nc = zeros(trail,1); % recording best number of centers, i.e. nc = (s+d,s)

for i=1:trail
    %% Generate the training and test samples
    tempI = randperm(DataSize);
    TrainI = tempI(1:m0); % index set of training samples
    ValidI = tempI(m0+1:m0+m1); % index set of validation samples
    TestI = tempI(m0+m1+1:DataSize); % index set of testing samples
    % Training samples
    xtr = dataset(TrainI,1:d-1);
    ytr = dataset(TrainI,d);
    % Validation samples
    xvalid = dataset(ValidI,1:d-1);
    yvalid = dataset(ValidI,d);
    % Test samples
    xte = dataset(TestI,1:d-1);
    yte = dataset(TestI,d);
    
    max_s = min(ceil((m0/log(m0))^(1/(d-1))),10); % the maximal bound for s in theory, max_s = 2
    
    %% training the best parameters via validation    
    s = (1:max_s)'; % the candidates of s
    ValidErr = zeros(max_s,1); % validation error for each s
    temp_TrainTime = zeros(max_s,1); % training time for each s
    for j=1:max_s
        nc = min(nchoosek(s(j)+d-1,d-1),m0); % number of centers
        cx = xtr(1:nc,:); % strategy 1
        
        [~,~,ValidErr(j),temp_TrainTime(j),~] = FPC(xtr,ytr,xvalid,yvalid,s(j),cx);
        fprintf('trail:%d, s:%d,validerr:%f\n',i, s(j), ValidErr(j));
    end
    traintime(i) = sum(temp_TrainTime); % total training time
    
    bestj = find(ValidErr==min(ValidErr));
    best_s(i) = s(bestj(1)); % the best degree of polynomial parameters
    
    
    %% training and testing using the best parameter obtained via validation
    best_nc(i) = min(nchoosek(best_s(i)+d-1,d-1),m0); % the best number of centers
    best_cx = xtr(1:best_nc(i),:); %the best centers
    [beta,trainerr(i),testerr(i),~,testtime(i)]=FPC(xtr,ytr,xte,yte,best_s(i),best_cx);    
    
    fprintf('trail:%d, trainerr:%f, testerr:%f, traintime:%f, testtime:%f\n', ...
        i, trainerr(i), testerr(i), traintime(i), testtime(i));
end
fprintf('testerr:%f,trainerr:%f\n',(1-mean(testerr)),(1-mean(trainerr)));

fprintf('stdoftesterr:%f\n',std(testerr));

fprintf('traintime:%f, testtime:%f\n',mean(traintime),mean(testtime));

fprintf('best_s:%f, best_nc:%f\n',mean(best_s), mean(best_nc));