# # Loads a pretrained Hugging Face image classification model for facial expression recognition.

# from PIL import Image
# import torch
# import numpy as np
# from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# # NOTE: you may swap MODEL_NAME to another Hugging Face model if desired
# MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

# class EmotionModel:
#     def init(self, device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         # feature extractor and model are cached by transformers in ~/.cache/huggingface
#         self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
#         self.model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(self.device)
#         # labeling mapping (if available)
#         self.id2label = getattr(self.model.config, "id2label", None) or {
#             i: str(i) for i in range(self.model.config.num_labels)
#         }

#     def predict(self, pil_image):
#         # pil_image: PIL.Image
#         inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
#         # map to label:score dict sorted desc
#         result = {self.id2label.get(i, str(i)): float(scores[i]) for i in range(len(scores))}
#         return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))

from PIL import Image
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

class EmotionModel:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(self.device)
        self.id2label = getattr(self.model.config, "id2label", {i: str(i) for i in range(self.model.config.num_labels)})

    def predict(self, pil_image):
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        result = {self.id2label[i]: float(scores[i]) for i in range(len(scores))}
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
