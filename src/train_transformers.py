import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from prepare_data import load_model_and_tokenizer, prepare_dataloaders


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=2, dropout=0.1):
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


def save_checkpoint(model_state, clf_state, optimizer_state, epoch, acc, loss):
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"acc_{int(100 * acc)}_epoch_{epoch}_{timestamp}.pt"
    )

    torch.save(
        {
            "clf_state_dict": clf_state,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "epoch": epoch,
            "accuracy": acc,
            "loss": loss,
        },
        checkpoint_path,
    )

    return checkpoint_path


def evaluate_model(model, clf, data_loader, device, return_predictions=False):
    """Enhanced evaluation with detailed metrics"""
    model.eval()
    clf.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
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

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = total_loss / len(data_loader)
    if return_predictions:
        return accuracy, avg_loss, all_preds, all_labels
    return accuracy, avg_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    model, tokenizer = load_model_and_tokenizer()
    train_loader, val_loader, test_loader = prepare_dataloaders(tokenizer, mode="sql")

    clf = LinearClassifier(hidden_size=model.config.hidden_size)
    clf.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.to(device)
    model.to(device)

    freeze_model_parameters(model)

    # use a lower learning rate for the transformer
    optimizer = optim.AdamW(
        [
            {"params": clf.parameters(), "lr": 2e-3, "weight_decay": 1e-2},
            {
                "params": filter(lambda p: p.requires_grad, model.parameters()),
                "lr": 2e-5,
                "weight_decay": 1e-4,
            },
        ]
    )

    num_epochs = 5

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        total_loss = 0

        clf.train()
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

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

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on validation set
        val_acc, val_loss = evaluate_model(model, clf, val_loader, device)  # type: ignore

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")

        # Save best classifier weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = save_checkpoint(
                {k: v for k, v in model.state_dict().items() if v.requires_grad},
                clf.state_dict(),
                optimizer.state_dict(),
                epoch,
                val_acc,
                val_loss,
            )
            print(f"  New best model saved! Accuracy: {val_acc:.4f}")

    print("Training complete.")

    # Evaluate best model on test set
    if best_checkpoint_path:
        checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        clf.load_state_dict(checkpoint["clf_state_dict"])
        # Load unfrozen model parameters
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Final test evaluation with detailed metrics
    test_acc, test_loss, test_preds, test_labels = evaluate_model(  # type: ignore
        model, clf, test_loader, device, return_predictions=True
    )

    print("\nFinal Test Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
