from torchvision.models import vgg16 as model_vgg
from models.base_model import set_final_layer

def vgg16():
    model = model_vgg(pretrained=True)
    model = set_final_layer(model, 'vgg16')
    return model
