from train import train_model

results = []

# loss, val_loss, acc = train_model(lr=0.001, epochs=3, run_name="baseline")
# results.append({"experiment": "baseline (lr=0.001, 3ep)", "train_loss": loss, "val_loss": val_loss, "test_acc": acc})

# loss, val_loss, acc = train_model(lr=0.001, epochs=6, run_name="epochs6")
# results.append({"experiment": "epochs=6", "train_loss": loss, "val_loss": val_loss, "test_acc": acc})

# loss, val_loss, acc = train_model(lr=0.01, epochs=3, run_name="lr_high")
# results.append({"experiment": "lr=0.01", "train_loss": loss, "val_loss": val_loss, "test_acc": acc})

# loss, val_loss, acc = train_model(lr=0.0001, epochs=3, run_name="lr_low")
# results.append({"experiment": "lr=0.0001", "train_loss": loss, "val_loss": val_loss, "test_acc": acc})

# Experiment 5: Dropout to combat overfitting seen at epochs=6
loss, val_loss, acc = train_model(lr=0.001, epochs=6, dropout=0.3, run_name="epochs6_dropout")
results.append({"experiment": "epochs=6 + dropout=0.3", "train_loss": loss, "val_loss": val_loss, "test_acc": acc})

# Experiment 6: Slightly fewer epochs (sweet spot check)
loss, val_loss, acc = train_model(lr=0.001, epochs=4, run_name="epochs4")
results.append({"experiment": "epochs=4", "train_loss": loss, "val_loss": val_loss, "test_acc": acc})

print("\n--- Experiment Results ---")
for r in results:
    print(r)