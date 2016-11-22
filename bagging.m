function y_hat = bagging(x,y,test_x,N)
%bagging�㷨
%���������
%xΪѵ�����ݵ�����ֵ(iris_feature)
%yΪѵ�����ݵı�ǩ(iris_label)
%test_xΪ��֤�����ݵ�����ֵ
%NΪ���ɵ���������������Ĭ��Ϊ20

%���������
%y_hatΪ��֤������������ѵ��������ѵ�����ɵľ������еķ�����

%���δ����N����NĬ��Ϊ10
if nargin < 4
    N = 10;
end


%��ѵ�������ݽ���boostrap����ʹ��CART�㷨��decision����������ѵ�����˹����ظ�N��
n = size(x,1);
bg_trees = cell(N,1);%ѵ���õ�N����������
for i = 1:N
    train_index = randsample(1:n,n,true);%boosting
    train_data  = x(train_index,:);
    train_label = y(train_index);
    tree        = cart(train_data,train_label);
    bg_trees{i} = tree;
end

%ʹ��������������֤�����ݽ��з��࣬�õ�n��N�еı�ǩ����
box = [];
for j   = 1:N
    b   = classify(test_x,bg_trees{j});
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