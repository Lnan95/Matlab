function y_hat = rforest(x,y,test_x,N)
%���ɭ���㷨
%���������
%xΪѵ�����ݵ�����ֵ(iris_feature)
%yΪѵ�����ݵı�ǩ(iris_label)
%test_xΪ��֤��������ֵ
%NΪ������������
%���������
%y_hatΪ��֤������������ѵ��������ѵ�����ɵľ������еķ�����

%��ģ��Ĭ��Ϊ20
if nargin<4
    N = 10;
end

%��ѵ�������ݽ���boostrap����ʹ��CART�㷨��decision����������ѵ�����˹����ظ�N��
[n,m]    = size(x);
rf_trees = cell(N,1);
for i = 1:N
    train_index   = randsample(1:n,n,true);%boostrap
    feature_index = randsample(1:m,round(sqrt(m)));%���ѡ������
    train_data    = x(train_index,feature_index);
    train_label   = y(train_index);
    tree          = decision(train_data,train_label,feature_index);
    rf_trees{i}   = tree;
end

%ʹ��������������֤�����ݽ��з��࣬�õ�n��N�еı�ǩ����
box = [];
for j   = 1:N
    b   = classify(test_x,rf_trees{j});
    box = [box b];
end

%���������Ӷ�����ԭ�򣬵õ����ս��y_hat
test_n = size(test_x,1);
y_hat  = [];
for k  = 1:test_n
    table_y   = tabulate(box(k,:)); 
    [~,index] = max(cell2mat(table_y(:,2)));
    y_hat     = [y_hat; table_y(index,1)];
end
end

function tree = decision(x,y,rf_feature_index)
%����cart�㷨�ľ�����
%���������
%xΪ�������ݵ�����ֵ����iris_feature
%yΪ�������ݵı�ǩ����iris_label
%rf_feature_indexΪ���ɭ���㷨���������������ֵ��ԭ����ֵ�е�λ��
%���������
%treeΪ���νṹ�����ݣ������ž�����ÿ���ڵ����Ϣ



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
tree = struct('leaf',false,'cond','','left','','right','','result','');


%Ѱ���������������ŵķָ��
[feature_index,bound] = divide(x,y);


left_x  = x(x(:,feature_index)>bound,:);
left_y  = y(x(:,feature_index)>bound);
right_x = x(x(:,feature_index)<=bound,:);
right_y = y(x(:,feature_index)<=bound);


%��������������������ɭ�֣�����rf_feature_index���������ݵ�������������
%��֤֪������������ݵ��ĸ�����

tree.cond   = [rf_feature_index(feature_index) bound];
tree.left   = decision(left_x, left_y,rf_feature_index);
tree.right  = decision(right_x, right_y,rf_feature_index);

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
