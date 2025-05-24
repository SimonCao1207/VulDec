import torch
from sklearn.metrics import classification_report, confusion_matrix

from prepare_data import load_model_and_tokenizer, prepare_dataloaders
from train_transformers import LinearClassifier, evaluate_model

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    train_loader, val_loader, test_loader = prepare_dataloaders(tokenizer, mode="sql")

    clf = LinearClassifier(hidden_size=model.config.hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.to(device)
    model.to(device)

    best_checkpoint_path = "./checkpoints/best_model_acc_97_epoch_4.pt"  # Path to the best model checkpoint

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
