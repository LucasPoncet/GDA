import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.data.splits import splits
from src.data.synthetic import make_dataset
from src.graphs.ring import generate_ring_transfer_graph
from src.metrics.sensitivity import jacobian_sensitivity
from src.models.mpnn import MPNN
from src.models.readout import Readout
from src.train.loop import evaluate, train

device = "cuda" if torch.cuda.is_available() else "cpu"

config = yaml.safe_load(Path(sys.argv[1]).read_text())
config["date"] = datetime.now().isoformat(timespec="seconds")
config["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
config["device"] = device

run_dir = Path("runs") / f"{datetime.now():%Y-%m-%d_%H%M}_{config['task']}_{config['model']}"
run_dir.mkdir(parents=True)
(run_dir / "config.yaml").write_text(yaml.safe_dump(config))

distances = config["distances"]
results = {"distance": distances, "test_acc": [], "sensitivity": [], "history": []}

for r in distances:
    torch.manual_seed(config["seed"])
    graphs = make_dataset(generator=generate_ring_transfer_graph, n_samples=config["n_samples"], seed=config["seed"], classes=config["classes"], nodes=2 * r)
    train_loader, val_loader, test_loader = splits(graphs, config["split"], seed=config["seed"], batch_size=config["batch_size"])

    model = MPNN(input_dim=config["classes"], hidden_dim=config["hidden_dim"], num_layers=r, normalize=config["normalize"])
    readout = Readout(hidden_dim=config["hidden_dim"], num_classes=config["classes"])

    history = train(model=model, readout=readout, train_loader=train_loader, val_loader=val_loader, num_epochs=config["num_epochs"], lr=config["lr"], device=device)

    results["history"].append(history)
    results["test_acc"].append(evaluate(model=model, readout=readout, loader=test_loader, device=device))
    results["sensitivity"].append(jacobian_sensitivity(model=model.cpu(), data=test_loader.dataset[0])[-1])
    torch.save({"model": model.state_dict(), "readout": readout.state_dict()}, run_dir / f"model_r{r}.pt")

(run_dir / "results.json").write_text(json.dumps(results, indent=2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
ax1.plot(distances, results["test_acc"], marker="o")
ax1.axhline(1 / config["classes"], linestyle="--", color="gray", label="chance")
ax1.set_xlabel("distance r (= number of layers)")
ax1.set_ylabel("test accuracy")
ax1.legend(loc="center right")
ax1.grid(True)
ax2.semilogy(distances, results["sensitivity"], marker="o")
ax2.set_xlabel("distance r (= number of layers)")
ax2.set_ylabel("sensitivity (trained)")
ax2.grid(True)
fig.tight_layout()
fig.savefig(run_dir / "ring_mpnn.png", dpi=200, bbox_inches="tight")