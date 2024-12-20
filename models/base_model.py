import torch.nn as nn

def set_final_layer(model: nn.Module, model_name: str):
    number_of_class = 3

    if model_name == "alexnet":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, number_of_class)
        for param in model.features.parameters():
            param.requires_grad = False

    elif model_name == "resnet":
        model.fc = nn.Linear(model.fc.in_features, number_of_class)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == "vgg16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, number_of_class)
        for param in model.features.parameters():
            param.requires_grad = False

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
