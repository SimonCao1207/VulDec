import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from prepare_data import load_model_and_tokenizer, prepare_dataloaders


class TransformerClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels=2):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_embeds):
        # input_embeds: [batch_size, seq_len, hidden_size]
        # Use [CLS] token embedding (first token)
        cls_embeds = input_embeds[:, 0, :]  # [batch_size, hidden_size]
        logits = self.classifier(cls_embeds)
        return logits


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    train_loader, val_loader, test_loader = prepare_dataloaders(tokenizer, mode="sql")

    # Assume model.config.hidden_size exists
    clf = TransformerClassifier(base_model=model, hidden_size=model.config.hidden_size)
    clf.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.to(device)
    model.to(device)
    optimizer = optim.Adam(clf.parameters(), lr=2e-5)
    num_epochs = 3

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
                # Use last hidden state as input_embeds
                input_embeds = outputs.last_hidden_state

            logits = clf(input_embeds)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
        )

    print("Training complete.")
