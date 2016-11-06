function [which,Bound] = divide(train,label)
%用于特征的分割
[~,n] = size(train);
min_gini = 1;
for i = 1:n
    [bound,gini]=Gini(train(:,i),label);
    if gini<min_gini
        min_gini = gini;
        which = i;
        Bound = bound;
    end
end
end


function [bound,min_gini] = Gini(train,label)
%找到单一特征的最小gini值点
%y为样本标签，x为样本值，且都为1维，dot为分割点
xx = unique(train);%去数据中的结
xx=sort(xx);
min_gini = 1;
for i=1:length(xx)
    dot = xx(i);
    l1=label(train>dot);
    l2=label(train<=dot);
    ll1=length(l1);
    ll2=length(l2);
    %利用strcmp函数匹配标签名称
    p1 = sum(strcmp(l1,'Iris-setosa'))/ll1;
    p2 = sum(strcmp(l1,'Iris-virginica'))/ll1;
    p3 = 1-p1-p2;
    p12 = sum(strcmp(l2,'Iris-setosa'))/ll2;
    p22 = sum(strcmp(l2,'Iris-virginica'))/ll2;
    p32 = 1-p12-p22;
    gini = 2-(p1^2+p2^2+p3^2+p12^2+p22^2+p32^2);
    if gini < min_gini
        min_gini = gini;
        which = i;
        bound = dot;
    end
end
bound = (bound+xx(which+1))/2;
end

