import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, sur_prop = 1.0):
        super(Bottleneck, self).__init__()
        self.value = sur_prop
        self.survival_prop = torch.bernoulli(torch.tensor([sur_prop])).item()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        # if(self.value != 1.0):
        #     print("value: ",self.value)
        #     print("survival_prop: ", self.survival_prop)
        x = x*self.survival_prop + identity
        x=self.relu(x)
        
        return x

    def adjust_sur_prop(self,value):
        self.value = value
        self.survival_prop = torch.bernoulli(torch.tensor([value])).item()

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, sur_prop = 1.0):
        super(Block, self).__init__()
        self.survival_prop = torch.bernoulli(torch.tensor([sur_prop])).item()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x = x + identity
      x = self.relu(x)
      return x

    

        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3, sur_prop=1.0):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, sur_prop = sur_prop )
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2, sur_prop = sur_prop)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2, sur_prop = sur_prop)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2, sur_prop = sur_prop)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(512*ResBlock.expansion, out_features=2048,bias=True),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=num_classes,bias=True)
        )
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, sur_prop, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride, sur_prop = sur_prop))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)


    def adjust_sur_prop(self,value):
        for block in self.layer1:
            block.adjust_sur_prop(value)
        for block in self.layer2:
            block.adjust_sur_prop(value)
        for block in self.layer3:
            block.adjust_sur_prop(value)
        for block in self.layer4:
            block.adjust_sur_prop(value)
        


        
        
def ResNet50(num_classes, channels=3, sur_prop=1.0):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels, sur_prop=sur_prop)
    
def ResNet101(num_classes, channels=3, sur_prop=1.0):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels, sur_prop=sur_prop)

def ResNet152(num_classes, channels=3, sur_prop=1.0):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels, sur_prop=sur_prop)
