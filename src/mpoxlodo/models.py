"""Model registry: 12 backbones in three families behind one interface.

`load_backbone(key)` returns a torch.nn.Module whose forward(x) gives a
(batch, feat_dim) embedding, plus normalisation constants and input size.
Gated weights (MedSigLIP, DINOv3 on Hugging Face) need HF_TOKEN set, e.g.
from Kaggle secrets; PanDerm needs PANDERM_CKPT pointing to its weights.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


@dataclass
class ModelSpec:
    key: str
    family: str          # cnn | imagenet_vit | general_fm | biomedical_fm | derm_fm
    kind: str            # timm | open_clip | hf_siglip | hf_dinov3 | panderm
    ident: str
    img_size: int = 224
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    transformer: bool = False
    zero_shot: bool = False
    fallback: Optional[str] = None
    rev: int = 1         # bump when the loader changes: cached features are then re-extracted


REGISTRY: dict[str, ModelSpec] = {m.key: m for m in [
    ModelSpec("resnet50", "cnn", "timm", "resnet50.tv_in1k"),
    ModelSpec("densenet169", "cnn", "timm", "densenet169.tv_in1k"),
    ModelSpec("efficientnet_b0", "cnn", "timm", "efficientnet_b0.ra_in1k"),
    ModelSpec("convnext_tiny", "cnn", "timm", "convnext_tiny.fb_in1k"),
    ModelSpec("vit_b16_in21k", "imagenet_vit", "timm", "vit_base_patch16_224.augreg_in21k",
              mean=SIGLIP_MEAN, std=SIGLIP_STD, transformer=True),
    ModelSpec("swin_tiny", "imagenet_vit", "timm", "swin_tiny_patch4_window7_224.ms_in1k", transformer=True),
    ModelSpec("dinov2_b14", "general_fm", "timm", "vit_base_patch14_dinov2.lvd142m", transformer=True),
    ModelSpec("dinov3_b16", "general_fm", "timm", "vit_base_patch16_dinov3.lvd1689m", transformer=True,
              fallback="facebook/dinov3-vitb16-pretrain-lvd1689m"),
    ModelSpec("clip_b16", "general_fm", "open_clip", "ViT-B-16|openai", mean=CLIP_MEAN, std=CLIP_STD,
              transformer=True, zero_shot=True),
    ModelSpec("biomedclip", "biomedical_fm", "open_clip",
              "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|",
              mean=CLIP_MEAN, std=CLIP_STD, transformer=True, zero_shot=True),
    ModelSpec("medsiglip", "biomedical_fm", "hf_siglip", "google/medsiglip-448", img_size=448,
              mean=SIGLIP_MEAN, std=SIGLIP_STD, transformer=True, zero_shot=True),
    ModelSpec("panderm", "derm_fm", "panderm", "vit_large_patch16_224", transformer=True, rev=2),
]}

FAMILY_GROUP = {"cnn": "CNN", "imagenet_vit": "ImageNet ViT", "general_fm": "Foundation model",
                "biomedical_fm": "Foundation model", "derm_fm": "Foundation model"}


def _timm_backbone(spec: ModelSpec):
    import timm
    kw = {"pretrained": True, "num_classes": 0}
    if "dinov2" in spec.ident or "dinov3" in spec.ident:
        kw["img_size"] = spec.img_size
    return timm.create_model(spec.ident, **kw)


def load_backbone(key: str):
    """Return (module, spec, extras). extras holds text encoders for zero-shot."""
    import torch
    import torch.nn as nn

    spec = REGISTRY[key]
    extras = {}

    if spec.kind == "timm":
        try:
            m = _timm_backbone(spec)
        except Exception as e:  # DINOv3 may be missing from older timm
            if spec.fallback is None:
                raise
            print(f"[{key}] timm load failed ({e}); trying Hugging Face {spec.fallback}")
            from transformers import AutoModel

            hf = AutoModel.from_pretrained(spec.fallback, token=os.environ.get("HF_TOKEN"))

            class HFPool(nn.Module):
                def __init__(self, mdl):
                    super().__init__()
                    self.m = mdl

                def forward(self, x):
                    out = self.m(pixel_values=x)
                    return out.pooler_output if getattr(out, "pooler_output", None) is not None \
                        else out.last_hidden_state[:, 0]
            m = HFPool(hf)
        m.feat_dim = _probe_dim(m, spec)
        return m, spec, extras

    if spec.kind == "open_clip":
        import open_clip
        name, pretrained = spec.ident.split("|")
        if name.startswith("hf-hub:"):
            model, _, _ = open_clip.create_model_and_transforms(name)
            tok = open_clip.get_tokenizer(name)
        else:
            model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
            tok = open_clip.get_tokenizer(name)

        class Visual(nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.visual = mdl.visual

            def forward(self, x):
                return self.visual(x)
        vis = Visual(model)
        vis.feat_dim = _probe_dim(vis, spec)
        extras["encode_text"] = lambda texts, device: model.to(device).encode_text(tok(texts).to(device))
        extras["logit_scale"] = float(model.logit_scale.exp().item())
        return vis, spec, extras

    if spec.kind == "hf_siglip":
        from transformers import AutoModel, AutoTokenizer
        token = os.environ.get("HF_TOKEN")
        full = AutoModel.from_pretrained(spec.ident, token=token)
        tok = AutoTokenizer.from_pretrained(spec.ident, token=token)

        class SigVis(nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.vision_model = mdl.vision_model

            def forward(self, x):
                return self.vision_model(pixel_values=x).pooler_output
        vis = SigVis(full)
        vis.feat_dim = _probe_dim(vis, spec)

        def enc_text(texts, device):
            t = tok(texts, padding="max_length", return_tensors="pt").to(device)
            return full.to(device).get_text_features(**t)
        extras["encode_text"] = enc_text
        extras["logit_scale"] = float(full.logit_scale.exp().item())
        extras["image_head"] = lambda x, device: full.to(device).get_image_features(pixel_values=x)
        return vis, spec, extras

    if spec.kind == "panderm":
        import timm
        ckpt = os.environ.get("PANDERM_CKPT") or "/tmp/panderm/panderm_ll_data6_checkpoint-499.pth"
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError("Set PANDERM_CKPT to the PanDerm weights file "
                                    "(see https://github.com/SiyuanYan1/PanDerm).")
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        sd = convert_cae_state_dict(sd)
        has_ls = any(k.endswith("ls1.gamma") for k in sd)
        kw = {"init_values": 1e-5} if has_ls else {}
        if "fc_norm.weight" in sd and "norm.weight" not in sd:   # mean-pooled encoder
            kw["global_pool"] = "avg"
        m = timm.create_model(spec.ident, pretrained=False, num_classes=0, **kw)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"[panderm] layer scale: {has_ls}; missing {len(missing)} keys, unexpected {len(unexpected)}")
        if missing:
            print("[panderm] missing:", missing[:12])
        if unexpected:
            print("[panderm] unexpected (first 12):", unexpected[:12])
        if missing:
            raise RuntimeError("PanDerm weights do not fully match a timm ViT-L/16 after conversion; "
                               "see the key lists above.")
        m.feat_dim = _probe_dim(m, spec)
        return m, spec, extras

    raise ValueError(spec.kind)


def convert_cae_state_dict(sd: dict) -> dict:
    """Map a CAE/CAEv2/BEiT-style ViT encoder state dict (PanDerm) to timm names.

    - strips 'module.', 'encoder.', 'backbone.' prefixes and drops decoder,
      regressor, teacher and mask-token weights;
    - attn.q_bias + attn.v_bias -> attn.qkv.bias (the key bias is zero);
    - gamma_1 / gamma_2 (layer scale) -> ls1.gamma / ls2.gamma.
    """
    import torch
    drop = ("decoder", "regressor", "teacher", "mask_token", "lm_head", "rel_pos_bias", "projector",
            "head.", "momentum", "target")
    out = {}
    for k, v in sd.items():
        k2 = k
        for pre in ("module.", "encoder.", "backbone.", "student."):
            if k2.startswith(pre):
                k2 = k2[len(pre):]
        if any(k2.startswith(d) or ("." + d) in k2 for d in drop):
            continue
        k2 = k2.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma")
        out[k2] = v
    for k in [k for k in out if k.endswith("attn.q_bias")]:
        base = k[: -len("q_bias")]
        q, vb = out.pop(k), out.pop(base + "v_bias", None)
        if vb is None:
            vb = torch.zeros_like(q)
        out[base + "qkv.bias"] = torch.cat([q, torch.zeros_like(q), vb])
    return out


def _probe_dim(m, spec: ModelSpec) -> int:
    import torch
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, spec.img_size, spec.img_size))
    return int(out.shape[-1])


def param_count(m) -> int:
    return sum(p.numel() for p in m.parameters())
