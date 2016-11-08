function [no_,node,Index] = builttree(train,label)
	node = 1;
	[n,~] = size(train);  %����
	no_=1;
    Index=struct('tree_index',[1:1:n],'class',[],'is_leaf',[0],'no',[]);
    switch_ = 1;
    while(switch_)
		
		node=node+1;
        leaf_num = 0;
		tree_num = 2^(node-2); %��һ���ж��ٸ���,node=2ʱ��tree_num=1��node=3ʱ��tree_num=2
        for i = 1:tree_num 
            father_tree = i+2^(node-2)-1; %����һ��������������,��i=1��node=2ʱ��father_tree=1��node=3ʱ��father_tree=2
            index = Index(father_tree).tree_index;
            feature = unique(label(index));
            label_ = label(index);
            if length(index)>5 && length(feature)>1 %�Ƿ񹻴�
                data = train(index,:);
                [which,bound]=divide(data,label_);

                left_row=find(data(:,which)<=bound);
                right_row =find(data(:,which)>bound);

                Index_=struct('tree_index',[],'class',[],'is_leaf',[0],'no',[]);
                Index_.tree_index = left_row;
                Index = [Index Index_];
                Index_=struct('tree_index',[],'class',[],'is_leaf',[0],'no',[]);
                Index_.tree_index = right_row;
                Index = [Index Index_];

            else
                leaf_num = leaf_num+2;
                if Index(father_tree).is_leaf ==1
                    Index_=struct('tree_index',[index],'class',[],'is_leaf',[1],'no',[]); 
                    Index = [Index Index_ Index_];
                else
                    count=zeros(size(feature));
                    for j=1:length(feature)
                        count(j)=sum(strcmp(label_,feature(j)));
                    end
                    [~,f]=max(count);
                    Index_=struct('tree_index',[index],'class',[feature(f)],'is_leaf',[1],'no',[no_]);
                    Index = [Index Index_];
                    Index_=struct('tree_index',[index],'class',[],'is_leaf',[1],'no',[]);
                    Index = [Index Index_];
                    no_=no_+1;
                end
            end
        end
        if leaf_num>=2^(node-1)
            switch_=0;
        end
    end
    n=2^(node-1)-1;
    Index = Index(1:n);
end

function [which,Bound] = divide(train,label)
%���������ķָ�
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
%�ҵ���һ��������Сginiֵ��
%yΪ������ǩ��xΪ����ֵ���Ҷ�Ϊ1ά��dotΪ�ָ��
xx = unique(train);%ȥ�����еĽ�
xx=sort(xx);
min_gini = 1;
for i=1:length(xx)
    dot = xx(i);
    l1=label(train>dot);
    l2=label(train<=dot);
    ll1=length(l1);
    ll2=length(l2);
    %����strcmp����ƥ���ǩ����
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