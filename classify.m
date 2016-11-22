function y_hat = classify(test_x,tree)
%使用训练出来的树对test数据集进行分类
%输入变量：
%test_x为待分类的数据的特征值
%tree为利用decision函数训练出来的树结构数据
%输出变量：
%y_hat为验证集的数据在由训练集数据训练生成的决策树中的分类结果

n = size(test_x,1);

%将验证集数据中每个数据代入训练好的树中，得到相应的估计标签
y_hat = [];
for i = 1:n
    test_tree = tree;
    while ~test_tree.leaf
        feature_index = test_tree.cond(1,1);%cond中为一行二列的cell结构数据，
        bound         = test_tree.cond(1,2);%分别是分割特征和分割点
        if test_x(i, feature_index) >= bound
            test_tree = test_tree.left;
        else
            test_tree = test_tree.right;
        end
    end
    y_hat = [y_hat;test_tree.result];
end
end

