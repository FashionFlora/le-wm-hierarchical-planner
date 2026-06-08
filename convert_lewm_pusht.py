import json
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def without_hydra_meta(mapping):
    return {k: v for k, v in mapping.items() if not k.startswith("_")}


src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False,
    use_mask_token=False,
)


def mlp(key):
    return MLP(
        input_dim=cfg[key]["input_dim"],
        output_dim=cfg[key]["output_dim"],
        hidden_dim=cfg[key]["hidden_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )


model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**without_hydra_meta(cfg["predictor"])),
    action_encoder=Embedder(**without_hydra_meta(cfg["action_encoder"])),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
state_dict = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state_dict, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
print(out)
