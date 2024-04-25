from typing import Union, Tuple
import os

import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import torch
import timm
import re

import random
import numpy as np
import wandb

from src import NUMBER_OF_EMOT, MODELS_DIR, AFFECTNET_CAT_EMOT, PROCESSED_AFFECTNET_DIR
from src.models.POSTER_V2.PosterV2_8cls import *
from src.models.POSTER_V2.main import *



def seed_everything(seed):
    """Set seeds to allow reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def define_criterion(params:dict, label_smoothing:float = 0.0, datasplit:str = 'train', 
                     distillation:bool = False, device: torch.device = 'cuda') -> torch.nn.Module:
    """Define the criterion to use in the training of the model. The criterion is defined by the
    parameters in the dictionary params. The function returns the criterion to use in the training 
    of the model.
    Params:
        - params (dict): The dictionary with the parameters of the model. It should contain the 
            'criterion' key with the name of the criterion to use. It can be 'crossentropy'.
        - label_smoothing (float): The label smoothing to use in the crossentropy loss. It should be 
            a value between 0 and 1. Default is 0.0.
        - datasplit (str): The datasplit to use to calculate the label weights. It should be 'train' or 'val'.
        - distillation (bool): If True, the criterion is used for distillation. Default is False.
        - device (torch.device): The device to use for the model. It can be 'cuda' or 'cpu'.
    Returns:
        - torch.nn.Module: The criterion to use in the training of the model.
    """
    # Define criterion
    if 'criterion' not in params:
        criterion_name = 'crossentropy'
    else:
        criterion_name = params['criterion']
    if 'weighted_loss' not in params or distillation:
        weighted_loss = False
    else:
        weighted_loss = params['weighted_loss']

    if criterion_name.lower() == 'crossentropy': # Note that nn.CrossEntropyLoss is equivalent to the combination of LogSoftmax and NLLLoss.
        if weighted_loss:
            label_weights = torch.load(os.path.join(PROCESSED_AFFECTNET_DIR, 
                                                        'label_weights_'+datasplit+'.pt')).float().to(device)
            print(f"Using weighted loss with label weights: {label_weights}")
            criterion = nn.CrossEntropyLoss(reduction = 'mean', weight=label_weights, label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(reduction = 'mean', label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Invalid criterion parameter: {criterion_name}")
    
    return criterion



def define_optimizer(model:torch.nn.Module, optimizer_name:str, lr:float, momentum:float) -> torch.optim.Optimizer:
    """Define the optimizer to use in the training of the model. The optimizer is defined by the optimizer_name,
    the learning rate and the momentum. The function returns the optimizer to use in the training of the model.
    Params:
        - model (torch.nn.Module): The model to train.
        - optimizer_name (str): The name of the optimizer to use. It can be 'adam', 'adamw', 'rmsprop', 'sgd' or 'none'.
        - lr (float): The learning rate of the optimizer.
        - momentum (float): The momentum of the optimizer. It is only used if the optimizer is 'rmsprop' or 'sgd'.
    Returns:
        - torch.optim.Optimizer: The optimizer to use in the training of the model.
    """   
    # Define optimizer
    optimizer_name = optimizer_name.lower()
    lr = float(lr)
    if momentum != 'none':
        momentum = float(momentum)
        
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'none':
        optimizer = None
    else:
        raise ValueError(f"Invalid optimizer parameter: {optimizer_name}")
    
    return optimizer



def get_pred_distilled_model(model:torch.nn.Module, imgs:torch.Tensor, output_method:str) -> torch.Tensor:
    """Get the predictions of the distilled model. The function returns the 
    predictions of the distilled model based on the output output method chosen. 
    It can be chosen between 'class', 'distill' or 'both' (sum between 'distill' and 'class' logits)."""
    pred, pred_dist = model(imgs)
    if output_method == "class":
        return pred
    elif output_method == "distill":
        return pred_dist
    elif output_method == "both":
        return pred + pred_dist
    else:
        raise ValueError(f"Invalid embedding method: {output_method}")



def get_distributions(output:torch.Tensor) -> torch.Tensor:
    """Get the distributions from the output of the model. The input is the output of the model in logits.
        Params:
            - output (torch.Tensor): The output of the model in logits.
        Returns:
            - torch.Tensor: The distributions of the predictions."""
    return F.softmax(output, dim=1)



def get_predictions(output:torch.Tensor) -> list:
    """Get the predictions from the output of the model. The input is the output of the model in logits. 
    The function returns a list with the labels of the predictions as a string.
    Params:
        - output (torch.Tensor): The output of the model in logits.
    Returns:
        - list: The list of the labels of the predictions.
    """
    distrib = F.softmax(output, dim=1)
    label_indices = torch.argmax(distrib, dim=1).cpu().numpy()
    labels = [AFFECTNET_CAT_EMOT[i] for i in label_indices]
    return labels



class RearrangeLayer(nn.Module):
    """Rearrange output neurons according to the order provided. It is done for the second dimension 
    (dim=1) where the emotions are, as the first dimension is the batch size. The order is a tensor 
    with the desired order of the emotions."""
    def __init__(self, order):
        super().__init__()
        self.order = order
    def forward(self, x):
        return torch.index_select(x, 1, self.order)  
    


def resnet34(pretrained:bool = True, weights:str = "none") -> torch.nn.Module:
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 2048"""
    if pretrained: # equivalent to ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet34(weights = "DEFAULT")
    else:
        model = models.resnet34(weights = None)
    model.fc = nn.Linear(512, NUMBER_OF_EMOT)

    if weights != "none":
        model.load_state_dict(weights)
        
    return model



def resnet50(pretrained:bool = True, weights:str = "none") -> torch.nn.Module:
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 2048"""
    if pretrained: # equivalent to ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights = "DEFAULT")
    else:
        model = models.resnet50(weights = None)
    model.fc = nn.Linear(2048, NUMBER_OF_EMOT)
    if weights != "none":
        model.load_state_dict(weights)
        
    return model



def resnet101(pretrained:bool = True, weights:str = "none") -> torch.nn.Module:
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 2048"""
    if pretrained: # equivalent to ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet101(weights = "DEFAULT")
    else:
        model = models.resnet101(weights = None)
    model.fc = nn.Linear(2048, NUMBER_OF_EMOT)

    if weights != "none":
        model.load_state_dict(weights)
        
    return model



def resnext50_32x4d (pretrained:bool = True) -> torch.nn.Module:
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 2048"""
    if pretrained:
        model = models.resnext50_32x4d(weights = "DEFAULT") # equivalent to ResNeXt50_32x4d_Weights.IMAGENET1K_V1
    else:
        model = models.resnext50_32x4d(weights = None)
    model.fc = nn.Linear(2048, NUMBER_OF_EMOT)
    return model


def poster_v2(weights:str = "none") -> torch.nn.Module:
    """Input size is defined as 224x224x3, so the flattened tensor after all convolutional layers is 1000"""
    
    # create model
    model = pyramid_trans_expr2(img_size=224, num_classes=8)
    if weights.lower() == "none":
        pass
    elif weights.lower() == "affectnet_cat_emot":
        weights_path = os.path.join(MODELS_DIR, "Poster_V2", "affectnet-8-model_best.pth")
        weights = torch.load(weights_path)
        model = load_pretrained_weights(model, weights)
    else:
        raise ValueError(f"Invalid weights parameter: {weights}")
    return model



def efficientnet_b0(device:torch.device, pretrained:bool = True, weights:str = "none") -> torch.nn.Module:
    """Create a EfficientNetB0 model with the specified weights. The model is created with the specified weights.
    The expected input is a tensor of [B, 3, 224, 224]. So the flattened tensor after all convolutional layers is 1280.
    The last layer is a linear layer with 8 outputs that is reorganized to match the standard AffectNet order using the 
    RearrangeLayer class.
    Source: https://arxiv.org/abs/1905.11946
    Params:
        - device (torch.device): The device to use for the model. It can be 'cuda' or 'cpu'.
        - pretrained (bool): If True, the model is created with pre-trained weights with imagenet. 
            If False, the model is created with random weights.
        - weights (str): The weights for the model. If "none", the model is created with 
            random weights. If "affectnet_cat_emot", the model is created with the weights 
            of the model trained on affectnet.
    Returns:
        - torch.nn.Module: The created EfficientNetB0 model.
    """
    if weights.lower() == "none":
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained = pretrained)
        model.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=NUMBER_OF_EMOT))

    elif weights.lower()  == "affectnet_cat_emot":
        weights_path = os.path.join(MODELS_DIR, "EfficientNetB0", "enet_b0_8_best_vgaf.pt")
        print("Weights obtained from: https://github.com/av-savchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/enet_b0_8_best_vgaf.pt")
        model = torch.load(weights_path)

        # To match the standard AffectNet order targets it is needed to rearrange the output of the model, as it is sorted alphabetically
        efficientnet_order = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        class_to_index = {class_name: i for i, class_name in enumerate(efficientnet_order)}
        # Create a tensor of the desired order
        order_tensor = torch.tensor([class_to_index[class_name] for class_name in AFFECTNET_CAT_EMOT]).to(device)
        # Add the rearrange layer to the model
        model.classifier = nn.Sequential(
            model.classifier,
            RearrangeLayer(order_tensor)
        )
    else:
        raise ValueError(f"Invalid weights parameter: {weights}")

    return model



def efficientnet_b2(device:torch.device, pretrained:bool = True, weights:str = "none") -> torch.nn.Module:
    """Create a EfficientNetB2 model with the specified weights. The model is created with the specified weights.
    The expected input is a tensor of [B, 3, 224, 224]. So the flattened tensor after all convolutional layers is 1408.
    The last layer is a linear layer with 8 outputs that is reorganized to match the standard AffectNet order using the 
    RearrangeLayer class.
    Source: https://arxiv.org/abs/1905.11946
    Params:
        - device (torch.device): The device to use for the model. It can be 'cuda' or 'cpu'.
        - pretrained (bool): If True, the model is created with pre-trained weights with imagenet. 
            If False, the model is created with random weights.
        - weights (str): The weights for the model. If "none", the model is created with 
            random weights. If "affectnet_cat_emot", the model is created with the weights 
            of the model trained on affectnet.
    Returns:
        - torch.nn.Module: The created EfficientNetB2 model.
    """
    if weights.lower() == "none":
        model = timm.create_model('tf_efficientnet_b2_ns', pretrained = pretrained)
        model.classifier=nn.Sequential(nn.Linear(in_features=1408, out_features=NUMBER_OF_EMOT))

    elif weights.lower()  == "affectnet_cat_emot":
        weights_path = os.path.join(MODELS_DIR,"EfficientNetB2", "enet_b2_8_best.pt")
        print("Weights obtained from: https://github.com/av-savchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/enet_b2_8_best.pt")
        model = torch.load(weights_path)

        # To match the standard AffectNet order targets it is needed to rearrange the output of the model, as it is sorted alphabetically
        efficientnet_order = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        class_to_index = {class_name: i for i, class_name in enumerate(efficientnet_order)}
        # Create a tensor of the desired order
        order_tensor = torch.tensor([class_to_index[class_name] for class_name in AFFECTNET_CAT_EMOT]).to(device)
        # Add the rearrange layer to the model
        model.classifier = nn.Sequential(
            model.classifier,
            RearrangeLayer(order_tensor)
        )

    else:
        raise ValueError(f"Invalid weights parameter: {weights}")

    return model
    


def ViT_base16(pretrained:bool = True, weights:str = "none") -> torch.nn.Module:
    """Create a Vision Transformer model with the base size and 16x16 patch size. 
    The model is created with the specified weights. The expected input is a tensor of [B, 3, 224, 224].
    Source: https://arxiv.org/abs/2010.11929
    Parameters:
        - pretrained (bool): If True, the model is created with pre-trained weights with imagenet. 
            If False, the model is created with random weights.
        - weights (str): The weights for the model. If "none", the model is created with 
            random weights. If "affectnet_cat_emot", the model is created with the weights 
            of the model trained on affectnet.
    Returns:
        - torch.nn.Module: The created ViT model.
    """
    if pretrained: # pretraining on imagenet
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes = NUMBER_OF_EMOT)
    else:
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes = NUMBER_OF_EMOT)
    if weights != "none":
        model.load_state_dict(weights)
    
    return model



class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output
    def close(self):
        self.hook.remove()



class DeiT_model(nn.Module):
    def __init__(self, base_model):
        super().__init__()  # Call to parent class's __init__ method
        self.base_model = base_model
        self.handle = Hook(self.base_model.head_dist)

    def forward(self, x):
        pred = self.base_model(x)
        pred_dist = self.handle.output
        return pred, pred_dist
    
    def clean_hook(self):
        self.handle.close()



def DeiT (size:str = "tiny", pretrained:bool = True, weights:str = "none") -> torch.nn.Module:
    """Create a DeiT model with the specified size. The model is created with the specified weights.
    The expected input is a tensor of [B, 3, 224, 224].
    Source: https://arxiv.org/abs/2012.12877v2
    Parameters:
        - size (str): The size of the DeiT model. It can be "tiny", "small" or "base".
        - pretrained (bool): If True, the model is created with pre-trained weights with imagenet. 
            If False, the model is created with random weights.
        - weights (str): The weights for the model. If "none", the model is created with 
            random weights. If "affectnet_cat_emot", the model is created with the weights 
            of the model trained on affectnet.
    Returns:
        - torch.nn.Module: The created DeiT model.
    """
    if size == "tiny":
        model = timm.create_model('deit_tiny_distilled_patch16_224.fb_in1k', pretrained=pretrained, num_classes = NUMBER_OF_EMOT)
    elif size == "small":
        model = timm.create_model('deit_small_distilled_patch16_224.fb_in1k', pretrained=pretrained, num_classes = NUMBER_OF_EMOT)
    elif size == "base":
        model = timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=pretrained, num_classes = NUMBER_OF_EMOT)
    else:
        raise ValueError(f"Invalid DeiT model size: {size}")
    model = DeiT_model(model)

    if weights != "none":
        model.load_state_dict(weights)

    return model



def model_creation(arch_type:str, weights:Union[str, dict] = "none", device:torch.device = None) -> Tuple[torch.nn.Module, torch.device]:
    """
    Create a model with the specified architecture and weights. If weights is not specified, 
    the model will be created with random weights. If a pretraining is given, the model will 
    be created with it. But if the weights are given, the model will be created with them.
    Parameters:
        - arch_type (str): The architecture of the model. 
        - weights (str or dict, optional): The weights for the model. If a string is provided, 
            it should be either "imagenet" for pre-trained weights or "none" for random weights. 
            If a dict with the weights is provided (result of model.state_dict() of a torch model),
            it will be used as the weights for the model.
        - device (torch.device, optional): The device to use for the model. If not provided,
            it will use the GPU if available.
    Returns:
        - torch.nn.Module: The created model 
        - torch.device: The CUDA/CPU device that will be used to train/validate/inference the model
    """
    if device is None: # If device is not provided by the user, instanciate it
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        print(f'Using CUDA with {torch.cuda.device_count()} GPUs')
        print(f'Using CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}')

    if isinstance(weights, str): # If weights is a string
        if weights.lower() == "imagenet": # Download pre-trained weights
            pretrained = True
            weights = "none"
        elif weights.lower() == "affectnet_cat_emot": # Use downloaded pre-trained weights on affectnet
            pretrained = False
            weights = "affectnet_cat_emot"
        elif weights.lower() == "none": # Random weights
            pretrained = False
            weights = "none"
        else:
            raise ValueError(f"Invalid weights parameter: {weights}")
    else: # Weights are provided as dict
        pretrained = False

    # Create the model following the architecture specified in the parameters
    arch_type = arch_type.lower()

    print(f"Creating model with architecture: {arch_type}")
    print(f"Using pre-trained weights: {pretrained}")
    print(f"Using custom weights of type {type(weights)}: {weights if isinstance(weights, str) else ''}")
    
    if arch_type.startswith("resnet"): # ResNet architectures
        size = "".join(re.findall(r'\d+', arch_type))
        if size == "34":
            model = resnet34(pretrained = pretrained, weights = weights)
        if size == "50":
            model = resnet50(pretrained = pretrained, weights = weights)
        elif size == "101":
            model = resnet101(pretrained = pretrained, weights = weights)
        
    elif arch_type.startswith("resnext"): # ResNeXt architecture
        model = resnext50_32x4d(pretrained = True)

    elif arch_type.startswith("poster"): # POSTER architectures
        model = poster_v2(weights = weights)

    elif arch_type.startswith("efficientnet"): # EfficientNet architectures
        if arch_type.endswith("b0"):
            model = efficientnet_b0(device, pretrained = pretrained, weights = weights)
        elif arch_type.endswith("b2"):
            model = efficientnet_b2(device, pretrained = pretrained, weights = weights)
    
    elif arch_type.startswith("vit"):
        model = ViT_base16(pretrained = pretrained, weights = weights)

    elif arch_type.startswith("deit"):
        size = arch_type.split('_')[1]
        model = DeiT(size, pretrained = pretrained, weights = weights)
    else:
        raise ValueError(f"Invalid model architecture: {arch_type}")

    # Move model to GPU
    model.to(device)

    return model, device



def get_wandb_artifact(wandb_id:str, run:wandb.run = None, api:wandb.api = None) -> str:
    """Download the model weights from wandb. The function returns the path to the downloaded 
    artifact.
    Params:
        - wandb_id (str): The id of the wandb artifact to download.
        - run (wandb.run, optional): The run object to use to download the artifact. 
            If not provided, the api object should be provided.
        - api (wandb.api, optional): The api object to use to download the artifact. 
            If not provided, the run object should be provided.
    Returns:
        - str: The path to the downloaded artifact."""
    
    print(f'Using trained model: {wandb_id}')
    try: # Try to download the model weights from name
        if not wandb_id.startswith("armand-07/TFG Facial Emotion Recognition/model_"):
            full_wandb_id = "armand-07/TFG Facial Emotion Recognition/model_" + wandb_id + ":latest"
        else:
            full_wandb_id = wandb_id
    except: # Try to download the model weights from run id
        print(f'The introduced wandb_id is not the name: {full_wandb_id}, trying to find it with run id')
        api = wandb.Api()
        if not wandb_id.startswith("armand-07/TFG Facial Emotion Recognition/"):
            full_wandb_id = "armand-07/TFG Facial Emotion Recognition/" + wandb_id
        else: 
            full_wandb_id = wandb_id
        name = api.run(full_wandb_id).name
        full_wandb_id = "armand-07/TFG Facial Emotion Recognition/model_" + name + ":latest"
    
    if run is not None:
        artifact = run.use_artifact(full_wandb_id, type = "model")
    elif api is not None:
        artifact = api.artifact(full_wandb_id, type = "model")

    # Download artifact and load params
    artifact_dir = artifact.download()
    print(f'Artifact downloaded to: {artifact_dir}')

    return artifact_dir
