function y_hat = classify(test_x,tree)
%ʹ��ѵ������������test���ݼ����з���
%���������
%test_xΪ����������ݵ�����ֵ
%treeΪ����decision����ѵ�����������ṹ����
%���������
%y_hatΪ��֤������������ѵ��������ѵ�����ɵľ������еķ�����

n = size(test_x,1);

%����֤��������ÿ�����ݴ���ѵ���õ����У��õ���Ӧ�Ĺ��Ʊ�ǩ
y_hat = [];
for i = 1:n
    test_tree = tree;
    while ~test_tree.leaf
        feature_index = test_tree.cond(1,1);%cond��Ϊһ�ж��е�cell�ṹ���ݣ�
        bound         = test_tree.cond(1,2);%�ֱ��Ƿָ������ͷָ��
        if test_x(i, feature_index) >= bound
            test_tree = test_tree.left;
        else
            test_tree = test_tree.right;
        end
    end
    y_hat = [y_hat;test_tree.result];
end
end

