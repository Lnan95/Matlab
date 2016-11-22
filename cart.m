function tree = cart(x,y,rf_feature_index)
%��֦���cart�㷨�ľ�����
%���������
%xΪ�������ݵ�����ֵ����iris_feature
%yΪ�������ݵı�ǩ����iris_label
%rf_feature_indexΪ���ɭ���㷨���������������ֵ��ԭ����ֵ�е�λ��
%���������
%treeΪ���νṹ�����ݣ������ž�����ÿ���ڵ����Ϣ

%�ɱ��������ж��Ƿ���ʹ�����ɭ��,�����������3������rfģʽ
if nargin<3
    rf = 0;
else
    rf = 1;
end

%���ý������ѡȡ���ŵ�����
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
%ѡȡ�����������ȷ����ߵ���������Ϊ��֦�����
accuracy = zeros(6,1);
for i = 1:6
    y_hat = classify(x,subtree{i});
    accuracy(i) = sum(strcmp(y_hat,y));
end
[~,index] = max(accuracy);
tree = subtree{index};
end

function tree = decision(x,y,rf_feature_index)
%����cart�㷨�ľ�����
%���������
%xΪ�������ݵ�����ֵ����iris_feature
%yΪ�������ݵı�ǩ����iris_label
%rf_feature_indexΪ���ɭ���㷨���������������ֵ��ԭ����ֵ�е�λ��
%���������
%treeΪ���νṹ�����ݣ������ž�����ÿ���ڵ����Ϣ

%�ɱ��������ж��Ƿ���ʹ�����ɭ��,�����������3������rfģʽ
if nargin<3
    rf = 0;
else
    rf = 1;
end

%���������������5����ֹͣ������������Ľ��Ϊ�⼸�������г�����������
n = size(x,1);
if n <= 5
    tree.leaf   = true;
    table_y     = tabulate(y);
    [~,index]   = max(cell2mat(table_y(:,2)));
    tree.result = table_y(index,1);
    return;
end

%�������ÿ�����ݶ���ͬ��ֹͣ�������������ɭ���г��֣�
if size(unique(x,'rows'), 1) == 1
    tree.leaf   = true;
    table_y     = tabulate(y); 
    [~,index]   = max(cell2mat(table_y(:,2)));
    tree.result = table_y(index,1);
    return;
end

%������еı�ǩ����ͬ��ֹͣ���������Ϊ�ñ�ǩ
label_name  = unique(y);
label_count = length(label_name);
if label_count  == 1
    tree.leaf   = true;
    tree.result = y(1);
    return;
end

%����struct������ÿ����������
tree = struct('leaf',false,'cond','','left','','right','','result','','alpha','','Nt','','Rt','');


%Ѱ���������������ŵķָ��
[feature_index,bound] = divide(x,y);


left_x  = x(x(:,feature_index)>bound,:);
left_y  = y(x(:,feature_index)>bound);
right_x = x(x(:,feature_index)<=bound,:);
right_y = y(x(:,feature_index)<=bound);


%��������������������ɭ�֣�����rf_feature_index���������ݵ�������������
%��֤֪������������ݵ��ĸ�����
if rf
    tree.cond   = [rf_feature_index(feature_index) bound];
    table_y     = tabulate(y); 
    [~,index]   = max(cell2mat(table_y(:,2)));
    tree.result = table_y(index,1);%��ǰ����֦�������ı�ǩ��Ϊ��֦��׼��
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

%������СginiѰ���������ŵķָ����Ӻ���
function [which,BOUND] = divide(x,y)
%xΪ�������ݵ�����ֵ
%yΪ�������ݵı�ǩ

n        = size(x,2);
min_gini = inf;
for i = 1:n
    [bound,gini] = Gini(x(:,i),y);
    %Ѱ����С��giniֵ
    if gini < min_gini
        min_gini = gini;
        which    = i;
        BOUND    = bound;
    end
end
end

%Ѱ�ҵ�һ��������Сginiֵ����Ӻ���
function [bound,min_gini] = Gini(x,y)
%xΪһά����������ֵ
%yΪ�������ݵı�ǩ

uni_x = unique(x);%ȥ�����еĽ�
sort_x=sort(uni_x);
min_gini = inf;
for i=1:length(sort_x)
    dot = sort_x(i);
    l1=y(x>dot);
    l2=y(x<=dot);
    ll1=length(l1);
    ll2=length(l2);
    %����strcmp����ƥ���ǩ����
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
%���������������ֵȫ��ȶ����ֵ�gini=inf�����
if min_gini < inf
    bound = (bound+sort_x(which+1))/2;
else
    bound = inf;
end
end

function tree = alpha(test_x,test_y,tree,num)
%���ڼ�֦�ĺ���������ÿ��֦���ϵ�alphaֵ
%test_xΪ��֤��������ֵ
%test_yΪ��֤���ı�ǩ
%treeΪ����cart���ɵľ�����
%numΪ��֤���ĸ�������
global min_alpha
if tree.leaf
    correct  = strcmp(test_y,tree.result);
    tree.num = length(correct);
    tree.correct_num = sum(correct);
    %������֤�������ݲ��ڸ�Ҷ��㣬����RT�޷�����
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
    tree.Rt = rt*pt;                             %���㱾����������
    tree.RT = tree.right.RT+tree.left.RT;        %���㱾���������������
    tree.alpha = (tree.Rt - tree.RT)/(tree.Nt-1);%�����ֵ
    %Ѱ����С�Ħ�ֵ�������Ƹ�ȫ�ֱ���
    if tree.alpha < min_alpha
        min_alpha = tree.alpha;
    end
end
end


%Ѱ�ҵ�alpha��͵������������޼���������min_alpha
function tree = prune(tree)
%���������alphaֵ�ľ����������ô˺�����֦���ϵ�alpha�ҵ�������֦
%����ȫ�ֱ���min_alpha�Է������
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






