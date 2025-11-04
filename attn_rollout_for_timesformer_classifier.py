from transformers import TimesformerConfig, TimesformerForVideoClassification, AutoImageProcessor
from torch.utils.data import DataLoader
import torch
from peft import PeftModel
from dataset_dolos import DolosClassificationDataset
from utils import set_seed
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np

# matplotlib.use('Agg')


set_seed(42)

BATCH_SIZE = 1
EPOCHS = 20

timestamp = "2025-11-03_00-45"

processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600")
config = TimesformerConfig()
config.num_labels = 2

split_id = 2
epoch = 12

test_dataset = DolosClassificationDataset(f"data/test_fold{split_id}.csv", "data/video", processor)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

save_path_lora = f"out/{timestamp}/lora_timesformer_split{split_id}_epoch{epoch}"
save_path = f"out/{timestamp}/timesformer_split{split_id}_epoch{epoch}.pt"

model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600", config=config, 
                                            ignore_mismatched_sizes=True)
lora_model = PeftModel.from_pretrained(model, save_path_lora)
classifier_state_dict = torch.load(save_path, map_location="cpu")
lora_model.base_model.model.classifier.load_state_dict(classifier_state_dict)
lora_model = lora_model.to("cuda").to(torch.bfloat16)
lora_model.eval()

mean = torch.tensor(processor.image_mean).reshape(3, 1, 1)
std = torch.tensor(processor.image_std).reshape(3, 1, 1)

correct_test_preds = 0
all_test_preds = 0
for pixel_values, labels in test_dataloader:
    pixel_values = pixel_values.to(lora_model.device).to(torch.bfloat16)
    labels = labels.to(lora_model.device)

    with torch.inference_mode():
        output = lora_model(pixel_values=pixel_values, output_attentions=True)

    R = torch.eye(197, device=output.attentions[0].device).unsqueeze(0).repeat(8, 1, 1)
    for i in range(11, -1, -1):
        A = output.attentions[i].mean(1)
        A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0).repeat(8, 1, 1)
        A = A / A.sum(dim=-1, keepdim=True)
        R = R @ A
    res = R[:, 0, 1:]

    attn_map = res.cpu().numpy()
    attn_map = attn_map.reshape(8, 14, 14)
    attn_map = (attn_map - attn_map.min(axis=(1,2), keepdims=True)) / (attn_map.max(axis=(1,2), keepdims=True) - attn_map.min(axis=(1,2), keepdims=True))

    fig, axes = plt.subplots(1, 8, figsize=(20, 3))

    for i, ax in enumerate(axes.flat):
        attn_map_resized = cv2.resize(attn_map[i], (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_HOT)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        img = pixel_values[0, i].cpu() * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        img = np.dot(img, [0.2989, 0.5870, 0.1140])
        img = np.stack([img, img, img], axis=-1)
        img = 0.5 * img + 0.5 * heatmap
        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
    
    preds = torch.argmax(output.logits, dim=-1)
    correct_test_preds += torch.sum(preds == labels).item()
    all_test_preds += preds.size(0)
    print(torch.sum(preds == labels).item())
    plt.show()

best_test_accuracy = correct_test_preds / all_test_preds * 100

print(best_test_accuracy)
