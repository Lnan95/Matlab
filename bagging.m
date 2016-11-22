function y_hat = bagging(x,y,test_x,N)
%bagging算法
%输入变量：
%x为训练数据的特征值(iris_feature)
%y为训练数据的标签(iris_label)
%test_x为验证集数据的特征值
%N为生成的弱分类器数量，默认为20

%输出变量：
%y_hat为验证集的数据在由训练集数据训练生成的决策树中的分类结果

%如果未输入N，则N默认为10
if nargin < 4
    N = 10;
end


%对训练集数据进行boostrap，并使用CART算法（decision函数）进行训练，此过程重复N次
n = size(x,1);
bg_trees = cell(N,1);%训练得的N个弱分类器
for i = 1:N
    train_index = randsample(1:n,n,true);%boosting
    train_data  = x(train_index,:);
    train_label = y(train_index);
    tree        = cart(train_data,train_label);
    bg_trees{i} = tree;
end

%使用弱分类器对验证集数据进行分类，得到n行N列的标签集合
box = [];
for j   = 1:N
    b   = classify(test_x,bg_trees{j});
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