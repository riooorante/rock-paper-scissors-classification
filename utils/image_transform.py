from torchvision import transforms

def transform_data(model):
    if model == "alexnet":
        transforms_data = transforms.Compose([
                                transforms.Resize((227, 227)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    else:
        transforms_data = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transforms_data