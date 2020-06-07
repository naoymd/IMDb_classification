import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        d = residual.shape[2] - out.shape[2]
        out = residual[:, :, 0:-d] + out
        out = self.relu(out)
        # out += residual
        return out


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        d = residual.shape[2] - out.shape[2]
        out = residual[:, :, 0:-d] + out
        out = self.relu(out)
        # out += residual
        return out


class MSResNet(nn.Module):
    def __init__(self, input_size, args, layers=[1, 1, 1, 1]):
        hidden_size = args.hidden_size
        fix_length = args.fix_length
        self.inplanes3 = hidden_size*3
        self.inplanes5 = hidden_size*3
        self.inplanes7 = hidden_size*3
        super(MSResNet, self).__init__()

        self.conv = nn.Conv1d(
            fix_length, hidden_size*3, kernel_size=7, stride=2, padding=6, dilation=2, bias=False)
        self.bn = nn.BatchNorm1d(hidden_size*3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer3x3_1 = self._make_layer3(
            BasicBlock3x3, hidden_size*3, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(
            BasicBlock3x3, hidden_size*4, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(
            BasicBlock3x3, hidden_size*8, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, hidden_size*10, layers[3], stride=2)
        # maxplooing kernel size: 16, 11, 6
        self.maxpool3 = nn.AdaptiveAvgPool1d(fix_length)

        self.layer5x5_1 = self._make_layer5(
            BasicBlock5x5, hidden_size*3, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(
            BasicBlock5x5, hidden_size*4, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(
            BasicBlock5x5, hidden_size*8, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, hidden_size*10, layers[3], stride=2)
        self.maxpool5 = nn.AdaptiveAvgPool1d(fix_length)

        self.layer7x7_1 = self._make_layer7(
            BasicBlock7x7, hidden_size*3, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(
            BasicBlock7x7, hidden_size*4, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(
            BasicBlock7x7, hidden_size*8, layers[2], stride=2)
        # self.layer7x7_4 = self._make_layer7(BasicBlock7x7, hidden_size*10, layers[3], stride=2)
        self.maxpool7 = nn.AdaptiveAvgPool1d(fix_length)

        self.fc0 = nn.Linear(hidden_size*24, hidden_size*12)
        self.bn0 = nn.BatchNorm1d(hidden_size*12)
        self.fc1 = nn.Linear(hidden_size*12, hidden_size*6)
        self.bn1 = nn.BatchNorm1d(hidden_size*6)
        self.fc2 = nn.Linear(hidden_size*6, hidden_size*2)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*4, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=args.dropout)
        init.xavier_uniform_(self.fc0.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))
        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))
        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))
        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv(x0)
        x0 = self.bn(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        # x = self.layer3x3_4(x)
        x = self.maxpool3(x)

        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        # y = self.layer5x5_4(y)
        y = self.maxpool5(y)

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool7(z)

        out = torch.cat([x, y, z], dim=1)

        out = F.relu(self.bn0(self.fc0(out.permute(0, 2, 1))))
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        out = self.dropout(out)
        # print('c3d vid2vec', out.size())
        return out