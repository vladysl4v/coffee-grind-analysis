import timm


def get_vit_large(freeze_backbone=True):
    model = timm.create_model("vit_large_patch16_224", pretrained=True)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.reset_classifier(num_classes=1)

    return model
