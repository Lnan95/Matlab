function y_hat = rforest(x,y,test_x,N)
%随机森林算法
%输入变量：
%x为训练数据的特征值(iris_feature)
%y为训练数据的标签(iris_label)
%test_x为验证集的特征值
%N为弱分类器个数
%输出变量：
%y_hat为验证集的数据在由训练集数据训练生成的决策树中的分类结果

%弱模型默认为20
if nargin<4
    N = 10;
end

%对训练集数据进行boostrap，并使用CART算法（decision函数）进行训练，此过程重复N次
[n,m]    = size(x);
rf_trees = cell(N,1);
for i = 1:N
    train_index   = randsample(1:n,n,true);%boostrap
    feature_index = randsample(1:m,round(sqrt(m)));%随机选择特征
    train_data    = x(train_index,feature_index);
    train_label   = y(train_index);
    tree          = decision(train_data,train_label,feature_index);
    rf_trees{i}   = tree;
end

%使用弱分类器对验证集数据进行分类，得到n行N列的标签集合
box = [];
for j   = 1:N
    b   = classify(test_x,rf_trees{j});
    box = [box b];
end

%按少数服从多数的原则，得到最终结果y_hat
test_n = size(test_x,1);
y_hat  = [];
for k  = 1:test_n
    table_y   = tabulate(box(k,:)); 
    [~,index] = max(cell2mat(table_y(:,2)));
    y_hat     = [y_hat; table_y(index,1)];
end
end

function tree = decision(x,y,rf_feature_index)
%基于cart算法的决策树
%输入参数：
%x为输入数据的特征值，即iris_feature
%y为输入数据的标签，即iris_label
%rf_feature_index为随机森林算法中输入的数据特征值在原特征值中的位置
%输出参数：
%tree为树形结构的数据，储存着决策树每个节点的信息



%如果变量行数少于5个，停止生长，这颗树的结果为这几个变量中出现最多的特征
n = size(x,1);
if n <= 5
    tree.leaf   = true;
    table_y     = tabulate(y);
    [~,index]   = max(cell2mat(table_y(:,2)));
    tree.result = table_y(index,1);
    return;
end

%如果输入每例数据都相同，停止生长（易在随机森林中出现）
if size(unique(x,'rows'), 1) == 1
    tree.leaf   = true;
    table_y     = tabulate(y); 
    [~,index]   = max(cell2mat(table_y(:,2)));
    tree.result = table_y(index,1);
    return;
end

%如果所有的标签都相同，停止生长，结果为该标签
label_name  = unique(y);
label_count = length(label_name);
if label_count  == 1
    tree.leaf   = true;
    tree.result = y(1);
    return;
end

%利用struct储存树每个结点的数据
tree = struct('leaf',false,'cond','','left','','right','','result','');


%寻找输入数据中最优的分割点
[feature_index,bound] = divide(x,y);


left_x  = x(x(:,feature_index)>bound,:);
left_y  = y(x(:,feature_index)>bound);
right_x = x(x(:,feature_index)<=bound,:);
right_y = y(x(:,feature_index)<=bound);


%迭代生成树，如果是随机森林，利用rf_feature_index对输入数据的特征进行引索
%保证知道输入的是数据的哪个特征

tree.cond   = [rf_feature_index(feature_index) bound];
tree.left   = decision(left_x, left_y,rf_feature_index);
tree.right  = decision(right_x, right_y,rf_feature_index);

end

%基于最小gini寻找数据最优的分割点的子函数
function [which,BOUND] = divide(x,y)
%x为输入数据的特征值
%y为输入数据的标签

n        = size(x,2);
min_gini = inf;
for i = 1:n
    [bound,gini] = Gini(x(:,i),y);
    %寻找最小的gini值
    if gini < min_gini
        min_gini = gini;
        which    = i;
        BOUND    = bound;
    end
end
end

%寻找单一特征的最小gini值点的子函数
function [bound,min_gini] = Gini(x,y)
%x为一维的输入特征值
%y为输入数据的标签

uni_x = unique(x);%去数据中的结
sort_x=sort(uni_x);
min_gini = inf;
for i=1:length(sort_x)
    dot = sort_x(i);
    l1=y(x>dot);
    l2=y(x<=dot);
    ll1=length(l1);
    ll2=length(l2);
    %利用strcmp函数匹配标签名称
    p1 = sum(strcmp(l1,'Iris-setosa'))/ll1;
    p2 = sum(strcmp(l1,'Iris-virginica'))/ll1;
    p3 = 1-p1-p2;
    p12 = sum(strcmp(l2,'Iris-setosa'))/ll2;
    p22 = sum(strcmp(l2,'Iris-virginica'))/ll2;
    p32 = 1-p12-p22;
    gini = ll1/(ll1+ll2)*(1-p1^2-p2^2-p3^2)+ll2/(ll1+ll2)*(1-p12^2-p22^2-p32^2);
    if gini < min_gini
        min_gini = gini;
        which = i;
        bound = dot;
    end
end
%避免出现输入特征值全相等而出现的gini=inf的情况
if min_gini < inf
    bound = (bound+sort_x(which+1))/2;
else
    bound = inf;
end
end
