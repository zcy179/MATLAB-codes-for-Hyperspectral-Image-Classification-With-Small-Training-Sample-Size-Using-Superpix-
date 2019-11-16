function  Demo
%ZHENG C, WANG N, CUI J. Hyperspectral Image Classification With
% Small Training Sample Size Using Superpixel-Guided Training Sample
%Enlargement. IEEE Transactions on Geoscience and Remote Sensing,
% 2019: 57(10): 7307-7316.

clear
close all
%clc

%% data sets%%%%%%%%%%%%%%%%%%%%%%%%%%
%size:Indian(145*145) PaviaU(610*340) SalinasA(512*217) KSC(512*614)
dataNameSet={'Indian_pines_corrected','PaviaU','Salinas_corrected'};

%Superpixel numbers
SpNums=round([145*145/64 610*340/121 512*217/121 512*614/121]);
%SpNums=[300 1600 918 2500];

%lambda sets
LmdSets=[1e-3 1e-2 1e-2 1e-2 0.1 0.1;
    1e-3 1e-2 1e-2 1e-2 1e-2 1e-2;
    1e-4 1e-4 1e-4 1e-4 1e-4 1e-4];

expTimes=20;%20 Monte Carlo runs
Ps=[5 10 15 20 30 40];%training samples per class
lthP=length(Ps);

for nameNb=1:3
    numSuperpixels=SpNums(nameNb);
    %%
    dataName=dataNameSet{nameNb};
    load(dataName);%load data
    
    [row,col,dim]=size(data);
    nPixel=row*col;
    %% Convert to matrix and normalize%%%%%%%%
    X=zeros(dim,nPixel);
    js=1;
    for c=1:col
        for r=1:row
            x=reshape(data(r,c,:),dim,1);
            m=min(x);
            tmp=(x-m)/(max(x)-m);%Normalization
            X(:,js)=tmp;
            js=js+1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%  Superpixel segmentation
    fileName=['data\SpSegm' dataName num2str(numSuperpixels)];
    if exist([fileName '.mat'],'file')
        load(fileName) %load superpixel segmentation results
    else
        [Sp,nSp]=SuperpixelSegmentation(data,numSuperpixels);
        save(fileName,'Sp','nSp')%save superpixel segmentation results
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    for pth=1:lthP
        P=Ps(pth);%P is one of [5 10 15 20 30]
        fileName=sprintf('Result%s%d.txt',dataName(1:6),P);
        for expT=1:expTimes
            nClass=max(label(:));
            %% 划分训练样本和测试样本
            rng(expT*10,'twister');%随机数生成器
            mask=false(row,col);%已知点掩模
            nListTrn=zeros(nClass,1);%第类的训练样本数
            nListClass=zeros(nClass,1);%每类的样本总数
            idTst=[];
            labels=label;
            js=1;
            for c=1:nClass
                id=find(label==c);
                n=numel(id);
                if ~n,continue;end
                nListClass(js)=n;
                labels(id)=js;
                if P<1
                    ntrnc=max(round(P*n),1); %第c类训练样本数
                else
                    ntrnc=P;
                end
                if ntrnc>=n
                    ntrnc=15;
                end
                nListTrn(js)=ntrnc;
                id1=randperm(n,ntrnc);
                mask(id(id1))=true;%已知点掩模，mask(r,c)=true,则(r,c)点为已知点
                id(id1)=[];
                idTst=[idTst; id];
                js=js+1;
            end
            %%%%
            nClass=js-1;
            nListTrn(js:end)=[];
            nListClass(js:end)=[];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %idTrn0=find(mask);
            predictedLabel=zeros(row,col); %预测类别矩阵
            predictedLabel(mask)=labels(mask);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            tic
            %% 首先将超像素内只包含一类训练样本的超像素识别为该类
            SpInfo.unrecg=true(nSp,1);%记录超像素是否已经识别
            SpInfo.gIdx=cell(nSp,1);%存储各超像素的索引
            SpInfo.ntp=zeros(nSp,1,'uint16');%各超像素中包含的训练样本类别数
            SpInfo.types=cell(nSp,1);%各超像素中包含的训练样本类别
            for t=1:nSp
                idt= find(Sp==t & labels);%
                if isempty(idt)%该超像素不用识别
                    SpInfo.unrecg(t)=false;
                    SpInfo.gIdx{t}=[];
                    continue;
                end
                %查看其中是否包含已知类别数
                id1=find(mask(idt));
                ns=numel(id1);
                if ns %其中包含训练样本
                    lablei=labels(idt(id1));
                    types=unique(lablei);
                    ntp=numel(types);
                    if ntp==1 %类别数为1——仅仅包含一类训练样本
                        %将该超像素识别为该训练样本类
                        predictedLabel(idt)=types;
                        SpInfo.unrecg(t)=false;
                        %将已识别超像素作为训练样本
                        mask(idt)=true;
                        continue;
                    end
                    % 记录该超像素信息
                    SpInfo.ntp(t)=ntp;
                    SpInfo.types{t}=types;
                end
                SpInfo.gIdx{t}=idt;
            end
            tm0=toc;
            %% 识别内有多类训练样本或无训练样本的超像素
            idTrn=find(mask);
            [I,J] = ind2sub([row,col],idTrn);
            %trnLabel=labels(idTrn);
            trnLabel=predictedLabel(idTrn);
            A=X(:,idTrn);%训练矩阵
            %构造测试矩阵，由超像素内包含多类训练样本
            % 或不包含训练样本的超像素的均值向量构成
            id=find((SpInfo.ntp>1 | SpInfo.ntp==0)&SpInfo.unrecg);
            nT=numel(id);
            Y=zeros(dim,nT);
            yTypes=cell(nT,1);
            It=zeros(nT,1);
            Jt=zeros(nT,1);
            for t=1:nT
                idt=SpInfo.gIdx{id(t)};%
                Y(:,t)=mean(X(:,idt),2);%第t个超像素体数据集的均值向量
                yTypes{t}=SpInfo.types{id(t)};%第t个超像素体包含的训练样本类别
                [r0,c0]=ind2sub([row,col],idt);%第t个超像素体的坐标
                It(t)=round(mean(r0));%第t个超像素体重心位置行坐标
                Jt(t)=round(mean(c0));%第t个超像素体重心位置纵坐标
            end
            %%
            tstLabel=labels(idTst);
            %ratio和1-ratio为谱距离、空间距离所占比重;
            lambda=LmdSets(nameNb,pth);
            %利用距离加权回归分类器，对其进行分类；
            %%%%%%%%%%%%%%%%直接法%%%%%%%%%%%%%%%%%%%%%%%%%
            tic
            predLabel=DWLRC(A,Y,trnLabel,I,J,It,Jt,yTypes,lambda);
            for t=1:nT
                idt=SpInfo.gIdx{id(t)};%
                predictedLabel(idt)=predLabel(t);
            end
            tm1=toc+tm0;
            %% 计算各类识别精度
            [OA1, AA1, K, IA1]=ClassifyAccuracy(tstLabel,predictedLabel(idTst));
            %[IA2,OA2,AA2]=ComputeAccuracy(predictedLabel(idTst),tstLabel,nClass,nListClass-nListTrn);
            disp([P expT lambda  OA1 AA1 K tm1])
            tmp=[P expT lambda [OA1 AA1 K IA1']*100 tm1];
            dlmwrite(fileName,tmp,'-append','delimiter','\t','precision','%.4f')
        end%end of for expT
    end%end of for P
end%end of for nameNb
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [OA, AA, K, IA]=ClassifyAccuracy(true_label,estim_label)
% function ClassifyAccuracy(true_label,estim_label)
% This function compute the confusion matrix and extract the OA, AA
% and the Kappa coefficient.
%http://kappa.chez-alice.fr/kappa_intro.htm

l=length(true_label);
nb_c=max(true_label);

%compute the confusion matrix
confu=zeros(nb_c);
for i=1:l
    confu(true_label(i),estim_label(i))= confu(true_label(i),estim_label(i))+1;
end

OA=trace(confu)/sum(confu(:)); %overall accuracy
IA=diag(confu)./sum(confu,2);  %class accuracy
IA(isnan(IA))=0;
number=size(IA,1);

AA=sum(IA)/number;
Po=OA;
Pe=(sum(confu)*sum(confu,2))/(sum(confu(:))^2);
K=(Po-Pe)/(1-Pe);%kappa coefficient
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function predictedLabel=DWLRC(A,Y,trnLabel,I,J,It,Jt,yTypes,lambda)
%距离加权线性回归分类器
% min||Aix-y||_2^2+lambda||Wx||_2^2;
% Ai'Aix-A'y+lambdaW'Wx=0 =>x=(Ai'Ai+lambdaW'W)\(A'y)
% so Aix-y=Ai*inv(Ai'*Ai+lambda*W'*W)*Ai'y-y

nClass=max(trnLabel);
nTst=numel(It);
predictedLabel=zeros(nTst,1);
parfor t=1:nTst
    r0=It(t);
    c0=Jt(t);
    y=Y(:,t);
    err0=inf;
    if isempty(yTypes{t})
        classArray=1:nClass;
        nc=nClass;
    else
        classArray=yTypes{t};
        nc=numel(classArray);
    end
    for k=1:nc
        c=classArray(k);
        id=trnLabel==c;
        Ac=A(:,id);
        Ic=I(id);
        Jc=J(id);
        nck=numel(Ic);
        %%计算加权矩阵
        d=(Ic-r0).^2+(Jc-c0).^2;%空间距离
        W=diag(lambda*d);
        %%
        x=(Ac'*Ac+W)\(Ac'*y);
        d=Ac*x-y;
        %err=d'*d/(x'*x);%% 等于(||Acx-y||/||x||)^2
        err=d'*d;
        if err<err0
            err0=err;
            predictedLabel(t)=c;
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [labels_t,numlabels]=SuperpixelSegmentation(data,numSuperpixels)

[nl, ns, nb] = size(data);
x = data;
x = reshape(x, nl*ns, nb);
x = x';

input_img = zeros(1, nl * ns * nb);
startpos = 1;
for i = 1 : nl
    for j = 1 : ns
        input_img(startpos : startpos + nb - 1) = data(i, j, :);
        startpos = startpos + nb;
    end
end


%% perform Regional Clustering

%numSuperpixels = 200;  % number of segments
compactness = 0.1; % compactness2 = 1-compactness, compactness*dxy+compactness2*dspectral
dist_type = 2; % 1:ED；2：SAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: All pixels are clustered， 2：exist un-clustered pixels
% labels:segment no of each pixel
% numlabels: actual number of segments
[labels, numlabels, ~, ~] = RCSPP(input_img, nl, ns, nb, numSuperpixels, compactness, dist_type, seg_all);
clear input_img;

labels_t = zeros(nl, ns, 'int32');
for i=1:nl
    for j=1:ns
        labels_t(i,j) = labels((i-1)*ns+j);
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%