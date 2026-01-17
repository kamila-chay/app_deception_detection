# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import torch
from peft import PeftModel
from sklearn.metrics import auc, f1_score, roc_curve
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    TimesformerConfig,
    TimesformerForVideoClassification,
)

from thesis.utils.dataset_dolos import DolosClassificationDataset
from thesis.utils.utils import set_seed

set_seed(42)

BATCH_SIZE = 72
EPOCHS = 20

# should be merged with the main file

timestamp = "2025-11-03_00-45"

processor = AutoImageProcessor.from_pretrained(
    "facebook/timesformer-base-finetuned-k600"
)
config = TimesformerConfig()
config.num_labels = 2

for split_id, epoch in ((1, 16), (2, 12), (3, 12)):
    print(f"Split {split_id}")
    print(f"Epoch {epoch + 1}")

    test_dataset = DolosClassificationDataset(
        f"thesis/data/test_fold{split_id}.csv", "thesis/data/video", processor
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    save_path_lora = (
        f"thesis/out/{timestamp}/lora_timesformer_split{split_id}_epoch{epoch}"
    )
    save_path = f"thesis/out/{timestamp}/timesformer_split{split_id}_epoch{epoch}.pt"

    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k600",
        config=config,
        ignore_mismatched_sizes=True,
    )
    lora_model = PeftModel.from_pretrained(model, save_path_lora)
    classifier_state_dict = torch.load(save_path, map_location="cpu")
    lora_model.base_model.model.classifier.load_state_dict(classifier_state_dict)
    lora_model = lora_model.to("cuda").to(torch.bfloat16)
    lora_model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    for pixel_values, labels in test_dataloader:
        pixel_values = pixel_values.to(lora_model.device).to(torch.bfloat16)
        labels = labels.to(lora_model.device)
        with torch.inference_mode():
            output = lora_model(pixel_values=pixel_values)

        preds = torch.argmax(output.logits, dim=-1)
        probs = torch.softmax(output.logits, dim=-1)[:, 1]

        all_preds.append(preds)
        all_probs.append(probs)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, 0).cpu().numpy()
    all_probs = torch.cat(all_probs, 0).to(torch.float32).cpu().numpy()
    all_labels = torch.cat(all_labels, 0).cpu().numpy()

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(all_labels, all_preds, pos_label=0)

    print(f"F1: {f1}")
    print(f"AUC: {roc_auc}")
