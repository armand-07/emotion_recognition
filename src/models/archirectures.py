import torchvision.models as models
import torch.nn as nn

from src import NUMBER_OF_EMOT


def resnet50(pretrained = True):
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 2048"""
    if pretrained: # equivalent to ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights = "DEFAULT")
    else:
        model = models.resnet50(weights = None)
    model.fc = nn.Linear(2048, NUMBER_OF_EMOT)
    return model

def resnext50_32x4d (pretrained = True):
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 2048"""
    if pretrained:
        model = models.resnext50_32x4d(weights = "DEFAULT") # equivalent to ResNeXt50_32x4d_Weights.IMAGENET1K_V1
    else:
        model = models.resnext50_32x4d(weights = None)
    model.fc = nn.Linear(2048, NUMBER_OF_EMOT)
    return model

def wide_resnet50_2 ():
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 2048"""
    model = models.wide_resnet50_2(pretrained = True) # equivalent to WideResNet50_2_Weights.IMAGENET1K_V1
    model.fc = nn.Linear(2048, NUMBER_OF_EMOT)
    return model

#resnext50_32x4d = models.resnext50_32x4d(weights='DEFAULT')
#wide_resnet50_2 = models.wide_resnet50_2(weights='DEFAULT')

def vgg16(pretrained = True):
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 1000"""
    model = models.vgg16(pretrained = pretrained) 
    model.classifier[6] = nn.Linear(4096, NUMBER_OF_EMOT)
    return model
    
#vgg16 = models.vgg16(weights='DEFAULT')
