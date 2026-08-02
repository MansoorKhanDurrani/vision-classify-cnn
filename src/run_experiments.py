from train import train_model

results = []

# Best config so far: lr=0.001, epochs=6, dropout=0.3, filters=(16,32,64)
# Experiment: Same config, but WITHOUT augmentation
loss, val_loss, acc = train_model(lr=0.001, epochs=6, dropout=0.3, use_augmentation=False, run_name="no_augmentation")
results.append({"experiment": "best config, no augmentation", "train_loss": loss, "val_loss": val_loss, "test_acc": acc})

print("\n--- Experiment Results ---")
for r in results:
    print(r)