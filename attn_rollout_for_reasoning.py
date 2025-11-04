from transformers import TimesformerConfig, TimesformerForVideoClassification, AutoImageProcessor
from torch.utils.data import DataLoader
import torch
from peft import PeftModel
from dataset_dolos import DolosClassificationDataset
from utils import set_seed
import matplotlib.pyplot as plt
from utils import overlay_attention, roll_out_attn_map
from transformers import pipeline

set_seed(42)

BATCH_SIZE = 1
EPOCHS = 20

timestamp = "2025-11-03_00-45"

processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600")
config = TimesformerConfig()
config.num_labels = 2

pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

for split_id, epoch in ((2, 12), (1, 16), (3, 12)):
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

    for pixel_values, labels in test_dataloader:
        pixel_values = pixel_values.to(lora_model.device).to(torch.bfloat16)
        labels = labels.to(lora_model.device)

        with torch.inference_mode():
            output = lora_model(pixel_values=pixel_values, output_attentions=True)

        attn_map = roll_out_attn_map([attention[1::2] for attention in output.attentions], 4, 14, 14)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(overlay_attention(attn_map, pixel_values[:, 1::2], processor, i))
            ax.axis('off')
        plt.savefig("last_overlays.png")

        pred = torch.argmax(output.logits, dim=-1)
        del output, attn_map
        if torch.sum(pred == labels).item():
            messages = [
                {
                "role": "user",
                "content": [
                    {"type": "image", "url": "./last_overlays.png"},
                    {"type": "text", "text": "In the image you can see consecutive frames from a video. Patches marked in yellow indicate that a machine learning model deemed them very important for a classification task. Red means they're somewhat important, black means not important. We know that the classification task pertained to human behavioral analysis so we can safely ignore any patches that are marked as important but located in the background etc. Please write a short summary about possible cues the classification model used. Those can differ between frames, e.g can be the same at the beginning and in the middle but differ at the end. An example summary would be \"At the begining some possible cues are visible around the right eybrow. In the middle and at the end of the recording there are some signs around the mouth and nose of the subject.\" Output the summary only, stick to what you see."},
                    ],
                },
            ]

            out = pipe(text=messages, max_new_tokens=50)
            print(out)

        else:
            print("NONE ========")