function tree = cart(x,y,rf_feature_index)
%剪枝后的cart算法的决策树
%输入参数：
%x为输入数据的特征值，即iris_feature
%y为输入数据的标签，即iris_label
%rf_feature_index为随机森林算法中输入的数据特征值在原特征值中的位置
%输出参数：
%tree为树形结构的数据，储存着决策树每个节点的信息

%由变量多少判断是否是使用随机森林,如果变量等于3，开启rf模式
if nargin<3
    rf = 0;
else
    rf = 1;
end

%利用交叉检验选取最优的子树
n            = size(x,1);
num          = n/6;
random_index = randsample(1:n,n);
box          = cell(6,1);
subtree      = cell(6,1);
for i = 1:6
    box{i} = random_index((i-1)*round(1/6*n)+1:i*round(1/6*n));
end
for i  = 1:6
    train_index = setdiff(random_index,box{i});
    test_x      = x(box{i},:);
    test_y      = y(box{i});
    train_x     = x(train_index,:);
    train_y     = y(train_index);
    if rf
        tree = decision(train_x,train_y,rf_feature_index);
        tree = alpha(test_x,test_y,tree,num);
        tree = prune(tree);
        subtree{i} = tree;
    else
        tree = decision(train_x,train_y);
        tree = alpha(test_x,test_y,tree,num);
        tree = prune(tree);
        subtree{i} = tree;
    end

end
%选取交叉检验中正确率最高的子树，作为减枝后的树
accuracy = zeros(6,1);
for i = 1:6
    y_hat = classify(x,subtree{i});
    accuracy(i) = sum(strcmp(y_hat,y));
end
[~,index] = max(accuracy);
tree = subtree{index};
end

function tree = decision(x,y,rf_feature_index)
%基于cart算法的决策树
%输入参数：
%x为输入数据的特征值，即iris_feature
%y为输入数据的标签，即iris_label
%rf_feature_index为随机森林算法中输入的数据特征值在原特征值中的位置
%输出参数：
%tree为树形结构的数据，储存着决策树每个节点的信息

%由变量多少判断是否是使用随机森林,如果变量等于3，开启rf模式
if nargin<3
    rf = 0;
else
    rf = 1;
end

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
tree = struct('leaf',false,'cond','','left','','right','','result','','alpha','','Nt','','Rt','');


%寻找输入数据中最优的分割点
[feature_index,bound] = divide(x,y);


left_x  = x(x(:,feature_index)>bound,:);
left_y  = y(x(:,feature_index)>bound);
right_x = x(x(:,feature_index)<=bound,:);
right_y = y(x(:,feature_index)<=bound);


%迭代生成树，如果是随机森林，利用rf_feature_index对输入数据的特征进行引索
%保证知道输入的是数据的哪个特征
if rf
    tree.cond   = [rf_feature_index(feature_index) bound];
    table_y     = tabulate(y); 
    [~,index]   = max(cell2mat(table_y(:,2)));
    tree.result = table_y(index,1);%提前计算枝干上最多的标签，为减枝作准备
    tree.left   = decision(left_x, left_y,rf_feature_index);
    tree.right  = decision(right_x, right_y,rf_feature_index);
else
    tree.cond   = [feature_index bound];
    table_y     = tabulate(y); 
    [~,index]   = max(cell2mat(table_y(:,2)));
    tree.result = table_y(index,1);
    tree.left   = decision(left_x, left_y);
    tree.right  = decision(right_x, right_y);
end
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

function tree = alpha(test_x,test_y,tree,num)
%用于剪枝的函数，计算每个枝干上的alpha值
%test_x为验证集的特征值
%test_y为验证集的标签
%tree为利用cart生成的决策树
%num为验证集的个例总数
global min_alpha
if tree.leaf
    correct  = strcmp(test_y,tree.result);
    tree.num = length(correct);
    tree.correct_num = sum(correct);
    %避免验证集的数据不在该叶结点，导致RT无法计算
    if tree.num == 0
        tree.RT = 0;
    else
        tree.RT = 1 - tree.correct_num/tree.num;
    end
    tree.alpha  = inf;
    tree.Nt     = 1;
else
    which  = tree.cond(1,1);
    bound  = tree.cond(1,2);
    l_test = test_x(test_x(:,which)>bound,:);
    r_test = test_x(test_x(:,which)<=bound,:);
    l_test_y   = test_y(test_x(:,which)>bound);
    r_test_y   = test_y(test_x(:,which)<=bound);
    tree.left  = alpha(l_test,l_test_y,tree.left,num);
    tree.right = alpha(r_test,r_test_y,tree.right,num);
    tree.num   = tree.right.num + tree.left.num;
    tree.Nt    = tree.right.Nt + tree.left.Nt ;
    tree.correct_num = tree.right.correct_num + tree.left.correct_num;
    rt = 1 - (tree.right.correct_num+tree.left.correct_num)/(tree.right.num+tree.left.num);
    pt = (tree.right.num+tree.left.num)/num;
    tree.Rt = rt*pt;                             %计算本结点的误判率
    tree.RT = tree.right.RT+tree.left.RT;        %计算本结点子树的误判率
    tree.alpha = (tree.Rt - tree.RT)/(tree.Nt-1);%计算α值
    %寻找最小的α值，并复制给全局变量
    if tree.alpha < min_alpha
        min_alpha = tree.alpha;
    end
end
end


%寻找到alpha最低的子树，将其修剪，并重置min_alpha
function tree = prune(tree)
%输入已算得alpha值的决策树，利用此函数将枝干上的alpha找到，并剪枝
%设置全局变量min_alpha以方便迭代
global min_alpha
if tree.leaf
    return
else
    if tree.alpha == min_alpha
        tree.leaf = 1;
        min_alpha = inf;
        return
    end
    tree.left  = prune(tree.left);
	tree.right = prune(tree.right);
end
end






