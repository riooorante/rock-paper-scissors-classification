from torchvision.models import resnet50 as model_resnet50
from models.base_model import set_final_layer

def resnet50():
    model = model_resnet50(pretrained=True)
    model = set_final_layer(model, "resnet")
    return model
