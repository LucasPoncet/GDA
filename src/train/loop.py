import torch
import torch.nn.functional as F

from src.metrics.accuracy import accuracy


def train(model, readout, train_loader, val_loader, num_epochs=100, lr=1e-3, device="cpu"):
    """Train the model and restore the weights with the best validation accuracy."""
    model.to(device)
    readout.to(device)

    params = list(model.parameters()) + list(readout.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        readout.train()
        train_loss, train_correct, train_graphs = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            embeddings = model(batch)
            logits = readout(embeddings[-1], batch.readout_mask)
            loss = F.cross_entropy(logits, batch.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute training metrics
            predictions = logits.argmax(dim=1)
            n = batch.num_graphs

            train_loss += loss.item() * n
            train_correct += (predictions == batch.y).sum().item()
            train_graphs += n

        # Measure performance on the validation set
        model.eval()
        readout.eval()
        val_loss, val_correct, val_graphs = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Forward pass
                embeddings = model(batch)
                logits = readout(embeddings[-1], batch.readout_mask)
                loss = F.cross_entropy(logits, batch.y)

                # Compute validation metrics
                predictions = logits.argmax(dim=1)
                n = batch.num_graphs

                val_loss += loss.item() * n
                val_correct += (predictions == batch.y).sum().item()
                val_graphs += n

        train_loss /= train_graphs
        train_acc = train_correct / train_graphs
        val_loss /= val_graphs
        val_acc = val_correct / val_graphs

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch} | train loss {train_loss}, acc {train_acc}"
            f" | val loss {val_loss}, acc {val_acc}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": {name: value.clone() for name, value in model.state_dict().items()},
                "readout": {name: value.clone() for name, value in readout.state_dict().items()},
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        readout.load_state_dict(best_state["readout"])

    return history