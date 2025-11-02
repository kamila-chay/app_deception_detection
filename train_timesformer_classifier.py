from transformers import TimesformerConfig, TimesformerForVideoClassification, AutoImageProcessor
from torch.utils.data import DataLoader, DistributedSampler
import torch
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from peft import get_peft_model, LoraConfig, TaskType
from dataset_dolos import DolosClassificationDataset
from utils import set_seed
from torch.distributed import get_world_size, get_rank

set_seed(42)

BATCH_SIZE = 64
GRAD_ACCU_STEPS = 4
EPOCHS = 30

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

ds_config = {
    "train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": 2,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "fp16": {
        "enabled": False
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True
    },
    "optimizer": {
        "type": "DeepSpeedCPUAdam",
        "params": {
            "lr": 5e-5
        }
    }
}



for split_id in range(1, 4):

    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600")
    config = TimesformerConfig()
    config.num_labels = 2
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600", config=config, 
                                                            ignore_mismatched_sizes=True)
    
    train_dolos_dataset = DolosClassificationDataset(f"data/train_fold{split_id}.csv", "data/video", processor)
    train_sampler = DistributedSampler(train_dolos_dataset, get_world_size(), get_rank())
    train_dataloader = DataLoader(train_dolos_dataset, batch_size=BATCH_SIZE // get_world_size() // GRAD_ACCU_STEPS, sampler=train_sampler)

    lora_model = get_peft_model(model, lora_config)
    lora_model.base_model.model.classifier.weight.requires_grad = True
    lora_model.base_model.model.classifier.bias.requires_grad = True

    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(name)

    total_steps = ceil(len(train_dataset) / DEFAULT_BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    optimizer = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    scheduler = WarmupCosineLR(optimizer, total_num_steps=total_steps, warmup_num_steps=warmup_steps)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=lora_model,
        optimizer=optimizer,
        config=ds_config
    )

    model_engine.train()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        with torch.enable_grad():
            for batch_x, batch_y in train_dataloader:
                batch_x = batch_x.to(model_engine.device)
                batch_y = batch_y.to(model_engine.device)
                output = model_engine(pixel_values=batch_x, labels=batch_y)
                model_engine.backward(output.loss)

                model_engine.step()

        save_path_lora = f"./lora_timesformer_split{split_id}_epoch{epoch}"
        save_path = f"./timesformer_split{split_id}_epoch{epoch}"
        lora_model.save_pretrained(save_path_lora) 
        torch.save(lora_model.base_model.model.classifier.state_dict(), save_path)