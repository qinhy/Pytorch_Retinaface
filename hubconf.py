dependencies = ["torch"]

import os
import torch

from models.retinaface import RetinaFace

CFG_RESNET50 = {
    "name": "Resnet50",
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256
}

RESNET50_URL = "https://drive.google.com/file/d/14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW/view"
RESNET50_NAME = "Resnet50_Final.pth"

def retinaface_resnet50(pretrained=True, **kwargs):
    model = RetinaFace(cfg=CFG_RESNET50, phase="test")
    model.eval()

    if pretrained:
        if not os.path.exists(RESNET50_NAME):
            import gdown
            gdown.download(RESNET50_URL, RESNET50_NAME, quiet=False, fuzzy=True)
        state = torch.load(RESNET50_NAME, map_location=torch.device("cpu"), weights_only=False)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)

    return model