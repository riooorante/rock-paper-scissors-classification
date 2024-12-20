from torchvision.models import resnet18 as model_resnet18
from models.base_model import set_final_layer

def resnet18():
    model = model_resnet18(pretrained=True)
    model = set_final_layer(model, 'resnet')
    return model
