import torch
from util.pos_embed import interpolate_pos_embed

import timm
assert timm.__version__ == "0.6.13"  # version check
from timm.models.layers import trunc_normal_


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    
    return model