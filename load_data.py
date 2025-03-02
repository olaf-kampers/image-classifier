import json
from torchvision.models.resnet import ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
class_idx = weights.meta["categories"]
with open('imagenet_classes.json', 'w') as f:
    json.dump(class_idx, f)