import yaml
import torch
from omegaconf import OmegaConf
from layoutenc.modules.taming.models.vqgan import VQModel
from layoutenc.models.cond_transformer_w_layoutenc import Net2NetTransformer
from collections import OrderedDict
import numpy as np

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None):
  model = VQModel(**config.model.params.first_stage_config.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in sd.items():
        if 'first_stage_model' in k:
            name = k[18:]  # remove `module.`nvidia
            new_state_dict[name] = v

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def load_transformer(config, ckpt_path=None):
    model = Net2NetTransformer(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()


def to_numpy(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  return x

