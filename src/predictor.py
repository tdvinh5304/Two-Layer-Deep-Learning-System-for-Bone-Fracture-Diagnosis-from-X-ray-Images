import torch
from typing import Dict
from PIL import Image
import tempfile
import os

from model_handler import ModelHandler


class FracturePrediction:
    """Unified prediction class for bone type + fracture detection."""

    def __init__(self, model_handler: ModelHandler):
        self.model_handler = model_handler
        self.device = model_handler.device
        print("FracturePrediction initialized")

    @torch.inference_mode()
    def analyze_image(self, image: Image.Image) -> Dict:
        """
        Perform full analysis: bone type => fracture detection.
        Returns a dict containing the exact keys required by Visualizer.
        """

        try:
            # =======================================================
            # STEP 0 — Save PIL image to a temporary PNG file
            # =======================================================
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png"
            ) as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path)

            # =======================================================
            # STEP 1 — Predict bone type
            # =======================================================
            print("Predicting bone type...")
            bone_conf_probs = self.model_handler.predict_bone_type(temp_path)

            predicted_bone = max(bone_conf_probs, key=bone_conf_probs.get)
            bone_confidence = bone_conf_probs[predicted_bone]

            print(f"Bone = {predicted_bone} ({bone_confidence:.2%})")

            # =======================================================
            # STEP 2 — Predict fracture for that bone
            # =======================================================
            print(f"Predicting fracture for {predicted_bone}...")

            frac_probs = self.model_handler.predict_fracture(
                temp_path,
                predicted_bone
            )

            fracture_probability = float(frac_probs["fracture"])
            has_fracture = (fracture_probability > 0.5)

            if has_fracture:
                fracture_status = (
                    f"FRACTURE DETECTED with {fracture_probability:.1%} confidence"
                )
            else:
                fracture_status = (
                    f"NO FRACTURE DETECTED with {(1 - fracture_probability):.1%} confidence"
                )

            print("Analysis done:", fracture_status)

            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # =======================================================
            # STEP 3 — Return EXACT structure required by UI
            # =======================================================
            return {
                "bone_type": predicted_bone,
                "bone_confidence": bone_confidence,
                "bone_confidences": bone_conf_probs,
                "fracture_probability": fracture_probability,
                "has_fracture": has_fracture,
                "fracture_status": fracture_status
            }

        except Exception as e:
            print("Error during analysis:", e)
            raise
    