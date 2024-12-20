from torchvision.models import alexnet as model_alexnet
from models.base_model import set_final_layer

def alexnet():
    model = model_alexnet(pretrained=True)
    model = set_final_layer(model, 'alexnet')
    return model
