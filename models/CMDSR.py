import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import os

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        input = input.view(input.shape[0],-1)
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            # out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat, kernel_size=3, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                m.append(BaseConv2d(in_channel=n_feat, out_channel=4*n_feat, kernel_size=kernel_size, stride=1, padding=(kernel_size//2), bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(BaseConv2d(in_channel=n_feat, out_channel=9*n_feat, kernel_size=kernel_size, stride=1, padding=(kernel_size//2), bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class BaseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, use_support_Mod=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.padding = padding

        self.use_support_Mod = use_support_Mod

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input, condition_feature=None):
        '''
        input_features [batch, 64, h, w]
        condition_features [batch, 1, in_channel=64, 1, 1]
        '''
        b, c, h, w = input.shape
        if c != self.in_channel:
            raise ValueError('Input channel is not equal with conv in_channel')
        if self.use_support_Mod == True and condition_feature != None:
            #[batch, out_channel, in_channel, self.kernel_size, self.kernel_size] = [batch, 64, 64, 3, 3]
            weight = self.weight.unsqueeze(0) * self.scale * condition_feature 
            weight =weight.view(b*self.in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            input = input.view(1, b*self.in_channel, h, w)
            bias = torch.repeat_interleave(self.bias, repeats=b, dim=0)
            out = F.conv2d(input,weight,bias=bias,stride=self.stride,padding=self.padding,groups=b)
            _, _, height, width = out.shape
            out = out.view(b, self.out_channel, height, width)
        else:
            out = F.conv2d(input,self.weight * self.scale,bias=self.bias,stride=self.stride,padding=self.padding)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ResBlock(nn.Module):
    '''Residual block with controllable residual connections
condition feature----------------
                   |            |
              modulation    modulation
                   |            |
              ---Conv—--ReLU--Conv-+--
                 |_________________|   
    '''

    def __init__(self, nf=64, use_support_Mod=False, residual_lr=1, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = BaseConv2d(in_channel=nf, out_channel=nf, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True, use_support_Mod=use_support_Mod)
        self.conv2 = BaseConv2d(in_channel=nf, out_channel=nf, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True, use_support_Mod=use_support_Mod)
        self.act = nn.ReLU(inplace=True)
        self.residual_lr = residual_lr
        self.use_support_Mod = use_support_Mod

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, input_features, condition_features=None):
        '''
        input_features [batch, 64, h, w]
        condition_features [n_conv_each_block, batch, 1, 64, 1, 1]
        '''
        if self.use_support_Mod:
            res = self.conv1(input_features, condition_features[0])
            res = self.act(res)
            res = self.conv2(res, condition_features[1])
        else:
            res = self.conv1(input_features, None)
            res = self.act(res)
            res = self.conv2(res, None)
        out = res*self.residual_lr + input_features
        return out
        
class SRResNet_10(nn.Module):
    def __init__(self, use_support_Mod=True, scale=4, input_channels=3, channels = 64, n_block =10, n_conv_each_block=2, residual_lr = 1, kernel_size = 3, conv_index='22'):
        super(SRResNet_10, self).__init__()
        # self.args = args
        self.scale = scale
        self.input_channels = input_channels
        self.channels = channels
        self.n_block = n_block
        self.n_conv_each_block = n_conv_each_block
        self.residual_lr = residual_lr
        self.use_support_Mod = use_support_Mod
        self.conv_index = conv_index
        
        self.input_conv = BaseConv2d(in_channel=input_channels, out_channel=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        layers= []
        for _ in range(n_block):
            layers.append(ResBlock(nf=channels, use_support_Mod=self.use_support_Mod, residual_lr=residual_lr, kernel_size=kernel_size))
        self.backbone = nn.Sequential(*layers)
        self.upsampler = Upsampler(scale, channels, act=False)
        self.output_conv = BaseConv2d(in_channel=channels, out_channel=input_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)

        self.modulations = Modulations(n_block =self.n_block, n_conv_each_block=self.n_conv_each_block, conv_index=self.conv_index, 
                            sr_in_channel=self.channels)

    def forward(self, x, condition_feature=None):
        '''
        x [task_size*support_size, 3, h, w]
        condition_features [task_size, 128, 1, 1]

        modulated_condition_features [n_block, n_conv_each_block, task_size, 64, 1, 1, 1]
        '''
        if self.use_support_Mod:
            b, _, h, w = x.shape
            condition_feature = self.modulations(condition_feature)#[n_block, n_conv_each_block, task_size, 1, 64, 1, 1]
            condition_feature = torch.repeat_interleave(condition_feature, repeats=b//condition_feature.shape[2], dim=2)#[n_block, n_conv_each_block, batch, 1, 64, 1, 1]
        x = self.input_conv(x)
        res = x
        
        for block_idx, module in enumerate(self.backbone):
            if self.use_support_Mod:
                x = module(x, condition_feature[block_idx])
            else:
                x = module(x)
        x = res * self.residual_lr + x

        x = self.upsampler(x)
        x = self.output_conv(x)  

        return x
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))


class ConditionNet(nn.Module):
    def __init__(self, n_block =10, n_conv_each_block=2, conv_index='22', sr_in_channel=64, support_size=10):
        super(ConditionNet, self).__init__()
        self.support_size = support_size
        self.n_block = n_block
        self.n_conv_each_block = n_conv_each_block
        self.n_modulation = n_block*n_conv_each_block
        self.conv_index = conv_index
        if conv_index == '22':
            self.condition_channel = 128
        elif self.conv_index == '54':
            self.condition_channel = 256
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        self.sr_in_channel=sr_in_channel

        self.condition = self.get_VGG_condition()
        initialize_weights(self.condition,0.1)

    def get_VGG_condition(self):
        
        cfg = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']
        if self.conv_index == '22':
            cfg_idx = cfg[:5]
        elif self.conv_index == '54':
            cfg_idx = cfg[:35]
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        return self._make_layers(cfg_idx)
    
    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3*self.support_size
        for v in cfg:
            if v == 'P':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def reset_support_size(self, support_size):
        self.support_size = support_size

    def forward(self, support_x):
        '''return task_size condition_features
        Input:
        For training
        support_x [task_size, support_size*3, h, w]
        For testing task_size = 1
        support_x [1, support_size*3, h, w]
        
        '''
        support_conditional_feature = self.condition(support_x) #[task_size, 128, h/2, w/2]

        _, _, h, w = support_conditional_feature.shape
        conditional_feature = F.avg_pool2d(support_conditional_feature, kernel_size=h, stride=w)#[task_size, 128, 1, 1]
        return conditional_feature


class Modulations(nn.Module):
    def __init__(self, n_block =10, n_conv_each_block=2, conv_index='22', sr_in_channel=64):
        super(Modulations, self).__init__()
        self.n_block = n_block
        self.n_conv_each_block = n_conv_each_block
        self.n_modulation = n_block*n_conv_each_block
        self.conv_index = conv_index
        if conv_index == '22':
            self.condition_channel = 128
        elif self.conv_index == '54':
            self.condition_channel = 256
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        self.sr_in_channel=sr_in_channel

        self.modulations = self.get_linear_modulations()
        initialize_weights(self.modulations,0.1)
    
    def get_linear_modulations(self):
        modules = []
        for _ in range(self.n_modulation):
            modules.append(EqualLinear(self.condition_channel, self.sr_in_channel, bias_init=1))
        
        return nn.Sequential(*modules)
    
    def forward(self, condition_feature):
        '''
        Input:
        For training
        condition_feature:[task_size, 128, 1, 1]
        For testing
        condition_feature:[1, 128, 1, 1]

        repeat n_block*2 condition_features [n_block*n_conv_each_block, task_size, 128, 1, 3, 3]
        for i in range n_block*2:
            EqualLinear modulation condition_features[i] [task_size, 1, 64, 1, 1]
        condition_features [n_block, n_conv_each_block, task_size, 1, 64, 1, 1]
        '''
        task_size, condition_channel, h, w = condition_feature.shape
        if condition_channel != self.condition_channel:
            raise ValueError('the shape of input condition_feature should be [task_size, condition_channel, h, w]')

        condition_weight = []
        repeat_support_feature = torch.repeat_interleave(condition_feature.unsqueeze(0), repeats=self.n_modulation, dim=0)#[n_block*2, task_size, 128, 1, 1]
        for idx, modulation in enumerate(self.modulations):
            cur_support_feature = repeat_support_feature[idx]
            reshape_condition_feature = modulation(cur_support_feature.permute(0, 2, 3, 1)).view(task_size, 1, self.sr_in_channel, 1, 1)
            condition_weight.append(reshape_condition_feature.unsqueeze(0))
        
        out_features = torch.cat(condition_weight, 0).to(condition_feature.device)
        out_features = out_features.view(self.n_block, self.n_conv_each_block, task_size, 1, self.sr_in_channel, 1, 1)
        
        return out_features

class CMDSR(nn.Module):
    def __init__(self,args=None):
        super(CMDSR, self).__init__()
        self.args = args

        self.scale = self.args.scale_factor
        self.use_support_Mod = self.args.use_support_Mod
        self.task_size = self.args.task_size
        self.support_size = self.args.support_size
        
        self.input_channels = self.args.input_channels
        self.kernel_size = self.args.kernel_size
        self.channels = self.args.channels
        self.n_block = self.args.n_block
        self.n_conv_each_block = self.args.n_conv_each_block
        self.residual_lr = self.args.residual_lr
        self.conv_index = self.args.conv_index
        
        self.sr_net = SRResNet_10(use_support_Mod=self.use_support_Mod, scale=self.scale, input_channels=self.input_channels, 
                            channels=self.channels, n_block=self.n_block, n_conv_each_block=self.n_conv_each_block, residual_lr = self.residual_lr, kernel_size = self.kernel_size)

        if self.args.use_pretrained_sr_net:
            self._load_sr_net()

        if self.use_support_Mod:
            self.condition_net = ConditionNet(n_block =self.n_block, n_conv_each_block=self.n_conv_each_block, conv_index=self.conv_index, 
                            sr_in_channel=self.channels, support_size=self.support_size)

    def _load_sr_net(self):
        if os.path.exists(self.args.pretrained_model_checkpoint_dir+self.args.pretrained_sr_net_path.split("/")[-1]):
            print('loading pretrained model : {}'.format(self.args.pretrained_model_checkpoint_dir+self.args.pretrained_sr_net_path.split("/")[-1]))
        else:
            raise ValueError('Please get the pretrained BaseNet')
        self.sr_net.load_state_dict(torch.load(self.args.pretrained_model_checkpoint_dir+self.args.pretrained_sr_net_path.split("/")[-1],map_location='cpu'), strict=True)
    
    def _load_pretrain_net(self):
        if os.path.exists(self.args.pretrained_model_checkpoint_dir+self.args.load_trained_model_path.split("/")[-1]):
            print('loading pretrained model : {}'.format(self.args.pretrained_model_checkpoint_dir+self.args.load_trained_model_path.split("/")[-1]))
        else:
            raise ValueError('Please get the pretrained CMDSR')
        self.load_state_dict(torch.load(self.args.pretrained_model_checkpoint_dir+self.args.load_trained_model_path.split("/")[-1],map_location='cpu'), strict=True)

    def reset_task_size(self, task_size = 8):
        if not self.use_support_Mod:
            raise ValueError('Can not reset task_size if use_support_Mod = False!!!')
        self.task_size = task_size
    
    def reset_support_size(self, support_size = 6):
        if not self.use_support_Mod:
            raise ValueError('Can not reset support_size if use_support_Mod = False!!!')
        self.support_size = support_size
        self.condition_net.reset_support_size(support_size)

    def forward(self, x, support_x=None):
        condition_weight = None
        if self.use_support_Mod:
            condition_weight = self.condition_net(support_x)#[task_size, 128, 1, 1]
        x = self.sr_net(x, condition_weight)
        return x
        
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

if __name__ == "__main__":
    x = torch.randn(48,3,2,2)
    z = torch.randn(48,3,8,8)
    support_x = torch.randn(6,3,2,3)
    support_x1 = torch.randn(1,64,2,3)
    net = CMDSR()
    net.reset_task_size(1)
    y = net(x,x)
    train_optimizer = torch.optim.Adam(net.condition_net.parameters(), lr = 1e-3)
    loss = F.l1_loss(z, y)
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()
    for k, v in net.named_parameters():
        print(k)
        print(v.shape)
    x_test = torch.randn(1,3,2,2)
    net.reset_task_size(task_size=1)
    y1 = net(x_test, support_x)
    print(x)
