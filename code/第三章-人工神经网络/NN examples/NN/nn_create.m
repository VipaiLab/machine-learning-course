function nn = nn_create(varargin)%varargin相当于c中argv[]
%结构体用于各种参数、方法选择设置
SIZE = varargin{1};
nn.keep_probability     =               1;
nn.size                 =               SIZE;
nn.depth                =               numel(SIZE);%numel当数（）中元素个数就好
nn.active_function      =               'sigmoid';
nn.output_function      =               'sigmoid';
nn.learning_rate        =               1.5;%学习率
nn.weight_decay         =               0;%为防止过拟合，weight_decay是放在正则项前面的一个系数
nn.cost                 =               [];
nn.encoder              =               0;%编码器，用于初始赋值
nn.sparsity             =               0.03;%用于上面的稀疏编码，可略
nn.beta                 =               3;%用于下面的objective_function，可略
nn.batch_normalization  =               0;
nn.grad_squared         =               0;
nn.r                    =               0
nn.optimization_method  =               'normal';
nn.objective_function   =               'MSE';%均方误差，计算E



for i = 2:length(varargin)%matlab数组从1开始，由于主函数输入时只输入一个参数，length（）=1，故这段代码暂时没用
    if strcmp('active function',varargin{i})
        nn.active_function = varargin{i+1};
    elseif strcmp('output function',varargin{i})
        nn.output_function = varargin{i+1};
    elseif strcmp('learning rate',varargin{i})
        nn.learning_rate = varargin{i+1};
    elseif strcmp('weight decay',varargin{i})
        nn.weight_decay = varargin{i+1};
    elseif strcmp('sparsity',varargin{i})
        nn.sparsity = varargin{i+1};
    elseif strcmp('beta',varargin{i})
        nn.weight_decay = varargin{i+1};
    elseif strcmp('batch normalization',varargin{i})
        nn.batch_normalization = varargin{i+1};
    elseif strcmp('optimization method',varargin{i})
        nn.optimization_method = varargin{i+1};
    elseif strcmp('objective function', varargin{i})
        nn.objective_function = varargin{i+1};
    elseif strcmp('weight decay', varargin{i})
        nn.weight_decay = varargin{i+1};
    elseif strcmp('keep probability',varargin{i})
        nn.keep_probability = varargin{i+1};
    end;
end;

if strcmp(nn.objective_function, 'Cross Entropy')%应该也没交叉检验，故不用看
    nn.output_function = 'softmax';
end;

for k = 1 : nn.depth-1
    width = nn.size(k);
    height = nn.size(k+1);
    %nn.W{k} = (rand(height, width) - 0.5) * 2 * sqrt(6 / (height + width + 1)) - sqrt(6 / (height + width + 1));
    
    nn.W{k} = 2*rand(height, width)/sqrt(width)-1/sqrt(width);%rand产生伪随机数矩阵，即W权重矩阵初始化
    if abs(nn.keep_probability-1)>0.001
        nn.WMask{k} = ones(height,width);
    end;
    %nn.W{k} = 2*rand(height, width)-1;
    %Xavier initialization 
    if strcmp(nn.active_function, 'relu')
        nn.b{k} = rand(height,1)+0.01;
    else
        nn.b{k} = 2*rand(height, 1)/sqrt(width)-1/sqrt(width);%b阈值的初始化
    end;
    
        
    %parameters for moments 
    if strcmp(nn.optimization_method,'Momentum')%以下都是解决梯度方向等问题的优化方法，可以略
        nn.vW{k} = zeros(height,width);
        nn.vb{k} = zeros(height,1);
    end; 
    if strcmp(nn.optimization_method,'AdaGrad')  ||strcmp(nn.optimization_method,'RMSProp') || strcmp(nn.optimization_method,'Adam')
        nn.rW{k} = zeros(height,width);
        nn.rb{k} = zeros(height,1);
    end;
    if strcmp(nn.optimization_method,'Adam')
        nn.sW{k} = zeros(height,width);
        nn.sb{k} = zeros(height,1);
    end; 
       %parameters for batch normalization.
    if nn.batch_normalization
        nn.E{k} = zeros(height,1);
        nn.S{k} = zeros(height,1);
        nn.Gamma{k} = 1;
        nn.Beta{k} = 0;
        if  strcmp(nn.optimization_method,'Momentum')
            nn.vGamma{k} = 1;
            nn.vBeta{k} = 0;
        end;
        if strcmp(nn.optimization_method,'AdaGrad')  ||strcmp(nn.optimization_method,'RMSProp') || strcmp(nn.optimization_method,'Adam')
            nn.rW{k} = zeros(height,width);
            nn.rb{k} = zeros(height,1);
            nn.rGamma{k} = 0;
            nn.rBeta{k} = 0;
        end;
        if  strcmp(nn.optimization_method,'Adam')
            nn.sGamma{k} = 1;
            nn.sBeta{k} = 0;
        end;
    
        nn.vecNum = 0;
    end;
    nn.W_grad{k} = zeros(height,width);
end
if  strcmp(nn.optimization_method,'Adam')
    nn.AdamTime = 0;
end;