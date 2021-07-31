data=csvread("Data.csv");
X=data(:,2:10);
Y=data(:,12);%the edited data has the label column at no. 12
C=1;
iter=[1,2,20];
w=zeros(9,1);
w1=w;w2=w;

idx = randperm(699)  ; %generating training and testing dataset randomly
Xtrain = X(idx(1:466),:) ; 
Ytrain=Y(idx(1:466),1);
Xtest= X(idx(467:end),:) ;
Ytest=Y(idx(467:end),1);

%Xtrain=X(1:466,:);
%Ytrain=Y(1:466,1);
%Xtest=X(467:699,:);
%Ytest=Y(467:699,1);

for j=1:3   %loop for all 3 cases of iter=1,2 and 10
  for t=1:iter(j)
    for i=1:466
      x=Xtrain(i,:);
      y=Ytrain(i,1);
      y_pred=sign(x*w);
      loss=max(0,1-y*(x*w));
      l(t,i)=loss;
      tau=loss/(norm(x))^2;%PA
      %tau=min(C,loss/(norm(x))^2);%PA-I
      %tau=loss/((norm(x))^2+1/(2*C));%PA-II
      w=w+tau*y*x';
    endfor  

endfor
  Ytr_pred=sign(Xtrain*w);%predictions for train data 
  gtrain=abs(Ytr_pred-Ytrain);%training accuracy for iter 1,2,10
  tr_accuracy_PA(j)=(1-nnz(gtrain)/size(gtrain,1))*100;
  
  Ytst_pred=sign(Xtest*w);%predictions for test data 
  g=abs(Ytst_pred-Ytest);
  tst_accuracy_PA(j)=(1-nnz(g)/size(g,1))*100;%accuracy for iter=1,2,10
  
  weight_PA(:,j)=w; %weight vector for iter=1,2,10
endfor
tr_accuracy_PA
tst_accuracy_PA



for j=1:3   %loop for all 3 cases of iter=1,2 and 10
  for t=1:iter(j)
    for i=1:466
      x=Xtrain(i,:);
      y=Ytrain(i,1);
      y_pred=sign(x*w1);
      loss=max(0,1-y*(x*w1));
      l(t,i)=loss;
      %tau=loss/(norm(x))^2;%PA
      tau1=min(C,loss/(norm(x))^2);%PA-I
      %tau=loss/((norm(x))^2+1/(2*C));%PA-II
      w1=w1+tau1*y*x';
    endfor  

endfor
  Ytr_pred=sign(Xtrain*w1);%predictions for train data 
  gtrain=abs(Ytr_pred-Ytrain);%training accuracy for iter 1,2,10
  tr_accuracy_PAI(j)=(1-nnz(gtrain)/size(gtrain,1))*100;
  
  Ytst_pred=sign(Xtest*w1);%predictions for test data 
  g=abs(Ytst_pred-Ytest);
  tst_accuracy_PAI(j)=(1-nnz(g)/size(g,1))*100;%accuracy for iter=1,2,10
  
  weight_PAI(:,j)=w1; %weight vector for iter=1,2,10
endfor
tr_accuracy_PAI
tst_accuracy_PAI



for j=1:3   %loop for all 3 cases of iter=1,2 and 10
  for t=1:iter(j)
    for i=1:466
      x=Xtrain(i,:);
      y=Ytrain(i,1);
      y_pred=sign(x*w2);
      loss=max(0,1-y*(x*w2));
      l(t,i)=loss;
      %tau=loss/(norm(x))^2;%PA
      %tau=min(C,loss/(norm(x))^2);%PA-I
      tau2=loss/((norm(x))^2+1/(2*C));%PA-II
      w2=w2+tau2*y*x';
    endfor  

endfor
  Ytr_pred=sign(Xtrain*w2);%predictions for train data 
  gtrain=abs(Ytr_pred-Ytrain);%training accuracy for iter 1,2,10
  tr_accuracy_PAII(j)=(1-nnz(gtrain)/size(gtrain,1))*100;
  
  Ytst_pred=sign(Xtest*w2);%predictions for test data 
  g=abs(Ytst_pred-Ytest);
  tst_accuracy_PAII(j)=(1-nnz(g)/size(g,1))*100;%accuracy for iter=1,2,10
  
  weight_PAII(:,j)=w2; %weight vector for iter=1,2,10
endfor
tr_accuracy_PAII
tst_accuracy_PAII