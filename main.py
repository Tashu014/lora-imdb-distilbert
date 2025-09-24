from transformers import TextClassificationPipeline
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = small_train.map(tokenize, batched=True)
test_dataset = small_test.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
samples = [
    "This movie was absolutely fantastic, I loved every moment!",
    "The plot was boring and the acting was terrible.",
    "It was okay, not the best but not the worst either."
]

print("=== Predictions BEFORE Fine-Tuning ===")
for text in samples:
    print(text, "->", pipe(text))

args = TrainingArguments(
    output_dir="lora-imdb-distilbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

print("=== Predictions AFTER Fine-Tuning ===")
for text in samples:
    print(text, "->", pipe(text))
