data=csvread("Data.csv");
X=data(:,2:10);
X=[X ones(699,1)];
Y=data(:,12);%the edited data has the label column at no. 12
C=1;
iter=[1,2,20];
w=zeros(10,1);
idx = randperm(699)  ; %generating training and testing dataset randomly
Xtrain = X(idx(1:466),:) ; 
Ytrain=Y(idx(1:466),1);
Xtest= X(idx(467:end),:) ;
Ytest=Y(idx(467:end),1);

%Xtrain=X(1:466,:);
%Ytrain=Y(1:466,1);
%Xtest=X(467:699,:);
%Ytest=Y(467:699,1);

choice=menu('update method',{'PA','PA1','PA2'});
iter=input ('Maximum iteration?')
  

  %loop for all 3 cases of iter=1,2 and 10
  switch(choice)
     case 1
       disp('PA');
      for t=1:iter
        for i=1:466
          x=Xtrain(i,:);
          y=Ytrain(i,1);
          y_pred=sign(x*w);
          loss=max(0,1-y*(x*w));
          l(t,i)=loss;
          tau=loss/(norm(x))^2;%PA
          %tau=min(C,loss/(norm(x))^2);%PA-I
          #tau=loss/((norm(x))^2+1/(2*C));%PA-II
          w=w+tau*y*x';
        endfor  
      endfor
      
    case 2
      disp('PA-I');
      for t=1:iter
        for i=1:466
          x=Xtrain(i,:);
          y=Ytrain(i,1);
          y_pred=sign(x*w);
          loss=max(0,1-y*(x*w));
          l(t,i)=loss;
          %tau=loss/(norm(x))^2;%PA
          tau=min(C,loss/(norm(x))^2);%PA-I
          %tau=loss/((norm(x))^2+1/(2*C));%PA-II
          w=w+tau*y*x';
        endfor  
      endfor
      
    case 3
      disp('PA-II');
      for t=1:iter
        for i=1:466
          x=Xtrain(i,:);
          y=Ytrain(i,1);
          y_pred=sign(x*w);
          loss=max(0,1-y*(x*w));
          l(t,i)=loss;
          %tau=loss/(norm(x))^2;%PA
          %tau=min(C,loss/(norm(x))^2);%PA-I
          tau=loss/((norm(x))^2+1/(2*C));%PA-II
          w=w+tau*y*x';
        endfor  
       endfor
       
  endswitch
  Ytr_pred=sign(Xtrain*w);%predictions for train data 
  gtrain=abs(Ytr_pred-Ytrain);%training accuracy for iter 1,2,10
  tr_accuracy=(1-nnz(gtrain)/size(gtrain,1))*100;
  
  Ytst_pred=sign(Xtest*w);%predictions for test data 
  g=abs(Ytst_pred-Ytest);
  tst_accuracy=(1-nnz(g)/size(g,1))*100;%accuracy for iter=1,2,10
  
  weight=w; %weight vector for iter=1,2,10
tr_accuracy
tst_accuracy
  

