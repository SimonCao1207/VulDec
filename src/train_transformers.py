import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from prepare_data import load_model_and_tokenizer, prepare_dataloaders


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=2, dropout=0.2):
        super().__init__()
        self.clf = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_embeds):
        # input_embeds: [batch_size, seq_len, hidden_size]
        # Use [CLS] token embedding (first token)
        cls_embeds = input_embeds[:, 0, :]  # [batch_size, hidden_size]
        cls_embeds = self.dropout(cls_embeds)
        logits = self.clf(cls_embeds)
        return logits


def freeze_model_parameters(model, n=2):
    for param in model.parameters():
        param.requires_grad = False
    try:
        # Unfreeze the last n layers of the encoder
        encoder = model.encoder if hasattr(model, "encoder") else model.transformer
        for layer in encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
    except AttributeError:
        print("Could not unfreeze last layers: model structure not recognized.")


def save_checkpoint(best_clf_state, acc):
    # Save checkpoint
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_clf_acc_{int(100 * acc)}.pt")
    torch.save(best_clf_state, checkpoint_path)


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    train_loader, val_loader, test_loader = prepare_dataloaders(tokenizer, mode="sql")

    clf = LinearClassifier(hidden_size=model.config.hidden_size)
    clf.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.to(device)
    model.to(device)

    freeze_model_parameters(model)

    # use a lower learning rate for the transformer
    optimizer = optim.Adam(
        [
            {"params": clf.parameters(), "lr": 1e-3},
            {
                "params": filter(lambda p: p.requires_grad, model.parameters()),
                "lr": 5e-5,
            },
        ]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )
    num_epochs = 20

    best_val_acc = 0.0
    best_clf_state = None

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                input_embeds = outputs.last_hidden_state

            logits = clf(input_embeds)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
        )

        # Evaluate on validation set
        model.eval()
        clf.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                input_embeds = outputs.last_hidden_state
                logits = clf(input_embeds)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0
        scheduler.step(val_acc)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Save best classifier weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_clf_state = clf.state_dict()
            save_checkpoint(best_clf_state, best_val_acc)

        clf.train()
        model.train()

    print("Training complete.")

    # Evaluate best model on test set
    if best_clf_state is not None:
        clf.load_state_dict(best_clf_state)
    model.eval()
    clf.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            input_embeds = outputs.last_hidden_state
            logits = clf(input_embeds)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total if total > 0 else 0
    print(f"Test Accuracy (best val): {test_acc:.4f}")
