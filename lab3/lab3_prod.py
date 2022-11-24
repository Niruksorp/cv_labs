import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.models import alexnet, resnet50, vgg16, AlexNet_Weights, ResNet50_Weights, VGG16_Weights

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_pred(prediction, name, weights):
    val, index = torch.topk(prediction, 5)
    print(name)
    for i in range(5):
        category_name = weights.meta["categories"][index[i]]
        print(f"{category_name}: {100 * val[i]:.1f}%")


def read_image(i):
    from torchvision.io import read_image
    return read_image(f"data/{i}.jpg")


def create_res_net50_model():
    from torchvision.models import resnet50, ResNet50_Weights
    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    return model


def create_model(weights, model):
    modell = model(weights=weights)
    return modell.eval()


def create_alex_net_model(weights, model):
    from torchvision.models import alexnet, ResNet50_Weights, AlexNet_Weights

    # Step 1: Initialize model with the best available weights
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    return model


def create_predict(weights, img, model):
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    return model(batch).squeeze(0).softmax(0)


def create_vgg16_model():
    from torchvision.models import vgg16, ResNet50_Weights, VGG16_Weights

    # Step 1: Initialize model with the best available weights
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_alex_net = create_model(AlexNet_Weights.IMAGENET1K_V1, alexnet)
    model_res_net = create_model(ResNet50_Weights.DEFAULT, resnet50)
    model_vgg_16 = create_model(VGG16_Weights.IMAGENET1K_V1, vgg16)
    for i in range(50):
        img = read_image(i+1)
        print("=========================")
        print(f"exp{i}")
        print("=========================")
        pred = create_predict(AlexNet_Weights.IMAGENET1K_V1, img, model_alex_net)
        show_pred(pred, "alex_net", AlexNet_Weights.IMAGENET1K_V1)
        pred = create_predict(ResNet50_Weights.DEFAULT, img, model_res_net)
        show_pred(pred, "res net 50", ResNet50_Weights.DEFAULT)
        pred = create_predict(VGG16_Weights.IMAGENET1K_V1, img, model_vgg_16)
        show_pred(pred, "vgg 16", VGG16_Weights.IMAGENET1K_V1)
