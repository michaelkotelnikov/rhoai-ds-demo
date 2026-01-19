# train_pipeline.py
from kfp.dsl import component, pipeline, Output, Metrics
from kfp.compiler import compiler


BASE_IMAGE = "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/pytorch:2025.2"
PKGS = [
    "datasets",
    "transformers"
]


@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def train_distilgpt2(metrics: Output[Metrics],
                     epochs: int = 50,
                     batch_size: int = 4,
                     lr: float = 5e-5,
                     max_length: int = 64,
                     device: str = "cpu"):
    """Single-step training that mirrors the notebook and logs metrics."""
    import time, datetime, pathlib, torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    pairs = [
        {"prompt": "Hello, my name is",        "completion": " OpenShift Bot."},
        {"prompt": "The capital of France is", "completion": " Paris."},
        {"prompt": "2 + 2 equals",             "completion": " 4."},
        {"prompt": "Roses are red,",           "completion": " michael is blue."},
        {"prompt": "GPU stands for",           "completion": " Graphics Processing Unit."},
    ]
    lines = [p["prompt"] + p["completion"] for p in pairs]

    data_dir = pathlib.Path("./hello-world-dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.txt"
    with train_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")

    dataset = load_dataset("text", data_files=str(train_path))

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tokenizer))
    DEVICE = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.to(DEVICE)

    def collate_fn(batch):
        return {k: torch.stack([example[k] for example in batch]) for k in batch[0]}

    loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    start = time.time()
    final_loss = 0.0
    model.train()
    for epoch in range(epochs):
        for b in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            outputs = model(**b, labels=b["input_ids"])
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            final_loss = loss.item()
        print(f"Epoch {epoch}: loss={final_loss:.4f}")

    elapsed = time.time() - start

    save_dir = pathlib.Path("./models")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    metrics.log_metric("train_time_sec", float(elapsed))
    metrics.log_metric("train_time_hhmmss", str(datetime.timedelta(seconds=int(elapsed))))
    metrics.log_metric("final_loss", float(final_loss))


@pipeline(name="distilgpt2-train-pipeline")
def train_metric_pipeline(epochs: int = 50,
                          batch_size: int = 4,
                          lr: float = 5e-5,
                          max_length: int = 64,
                          device: str = "cpu"):
    train_distilgpt2(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        device=device,
    )


compiler.Compiler().compile(
    pipeline_func=train_metric_pipeline,
    package_path="train_metric_pipeline.yaml"
)