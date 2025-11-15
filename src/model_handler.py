import os
import torch
import torch.nn as nn
import numpy as np

from torchvision.models import resnet50, inception_v3

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    Resize,
    EnsureType,
    Lambda
)


# ================================================================================================
# UTILITIES — TRAIN-PREPROCESS FOR C_MODEL
# ================================================================================================

def to_3ch_train(x):
    """Convert grayscale → 3 channels exactly like training."""
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


def build_transform_cmodel(img_size: int):
    """
    Preprocess EXACTLY like your c_model training pipeline.
    NO ImageNet normalization.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        EnsureType(dtype=np.float32),
        Lambda(func=to_3ch_train),
        ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        Resize((img_size, img_size)),
    ])


# ================================================================================================
# UTILITIES — PREPROCESS FOR FRACTURE MODELS
# ================================================================================================

def to_3ch_np(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[None, :, :]
    channels = x.shape[0]

    if channels == 1:
        return np.repeat(x, 3, axis=0)
    if channels >= 3:
        return x[:3, :, :]
    return np.concatenate([x, x[:1, :, :]], axis=0)


def build_transform_fracture(img_size: int):
    """
    Keep fracture model preprocess as before (it worked).
    Includes ImageNet normalization.
    """
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def normalize_fn(arr: np.ndarray):
        arr = arr.astype(np.float32)
        return (arr - imagenet_mean) / imagenet_std

    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Lambda(func=to_3ch_np),
        ScaleIntensityRange(
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resize((img_size, img_size)),
        Lambda(func=normalize_fn),
        EnsureType(dtype=np.float32),
    ])


# ================================================================================================
# MODEL HANDLER (FIXED)
# ================================================================================================

class ModelHandler:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bone_types = [
            "ELBOW", "FINGER", "FOREARM",
            "HAND", "HUMERUS", "SHOULDER", "WRIST"
        ]

        self.bone_model = None
        self.fracture_models = {}

    # --------------------------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------------------------

    def preprocess_cmodel(self, img_path: str, img_size: int):
        tfm = build_transform_cmodel(img_size)
        arr = tfm(img_path)
        return torch.tensor(arr).unsqueeze(0).to(self.device)

    def preprocess_fracture(self, img_path: str, img_size: int):
        tfm = build_transform_fracture(img_size)
        arr = tfm(img_path)
        return torch.tensor(arr).unsqueeze(0).to(self.device)

    # --------------------------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------------------------
    def load_models(self):

        # ------------------------------
        # 1) Load bone classifier (c_model)
        # ------------------------------
        bone_ckpt_path = os.path.join(self.models_dir, "c_model_resnet50.pth")
        if not os.path.exists(bone_ckpt_path):
            raise FileNotFoundError(f"Bone classifier not found: {bone_ckpt_path}")

        ckpt = torch.load(bone_ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)

        bone_model = resnet50(weights=None)

        # Replace final FC with 7 output classes
        nf = bone_model.fc.in_features
        bone_model.fc = nn.Linear(nf, 7)

        # Fix weight slicing only if needed
        if "fc.weight" in state_dict and state_dict["fc.weight"].shape[0] != 7:
            state_dict["fc.weight"] = state_dict["fc.weight"][:7, :]
            state_dict["fc.bias"] = state_dict["fc.bias"][:7]

        bone_model.load_state_dict(state_dict, strict=False)
        self.bone_model = bone_model.to(self.device).eval()

        # ------------------------------
        # 2) Load fracture models
        # ------------------------------
        for bone in self.bone_types:
            inception_path = os.path.join(self.models_dir, f"XR_{bone}_inception3_best.pth")
            resnet_path = os.path.join(self.models_dir, f"XR_{bone}_resnet_best.pth")

            if os.path.exists(inception_path):
                arch = "inception"
                ckpt_path = inception_path
                img_size = 299
            elif os.path.exists(resnet_path):
                arch = "resnet"
                ckpt_path = resnet_path
                img_size = 320
            else:
                continue

            ck = torch.load(ckpt_path, map_location=self.device)
            sd = ck.get("state_dict", ck)

            if arch == "resnet":
                m = resnet50(weights=None)
            else:
                m = inception_v3(weights=None, aux_logits=False)

            nf = m.fc.in_features
            m.fc = nn.Linear(nf, sd["fc.weight"].shape[0])

            m.load_state_dict(sd, strict=False)
            m = m.to(self.device).eval()

            self.fracture_models[bone] = {
                "model": m,
                "img_size": img_size
            }

    # --------------------------------------------------------------------------
    # PREDICT BONE TYPE
    # --------------------------------------------------------------------------
    @torch.inference_mode()
    def predict_bone_type(self, img_path: str):
        tensor = self.preprocess_cmodel(img_path, img_size=320)
        logits = self.bone_model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        return {bone: float(probs[i]) for i, bone in enumerate(self.bone_types)}

    # --------------------------------------------------------------------------
    # PREDICT FRACTURE
    # --------------------------------------------------------------------------
    @torch.inference_mode()
    def predict_fracture(self, img_path: str, bone: str):
        if bone not in self.fracture_models:
            return None

        entry = self.fracture_models[bone]
        model = entry["model"]
        img_size = entry["img_size"]

        tensor = self.preprocess_fracture(img_path, img_size)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        return {
            "normal": float(probs[0]),
            "fracture": float(probs[1])
        }
