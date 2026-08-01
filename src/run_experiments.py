from train import train_model

results = []

# Baseline recheck (with the reusable function, 3 epochs, lr=0.001)
loss, acc = train_model(lr=0.001, epochs=3, run_name="baseline")
results.append({"experiment": "baseline (lr=0.001, 3ep)", "final_loss": loss, "test_acc": acc})

# Experiment 1: More epochs
loss, acc = train_model(lr=0.001, epochs=6, run_name="epochs6")
results.append({"experiment": "epochs=6", "final_loss": loss, "test_acc": acc})

# Experiment 2: Higher learning rate
loss, acc = train_model(lr=0.01, epochs=3, run_name="lr_high")
results.append({"experiment": "lr=0.01", "final_loss": loss, "test_acc": acc})

# Experiment 3: Lower learning rate
loss, acc = train_model(lr=0.0001, epochs=3, run_name="lr_low")
results.append({"experiment": "lr=0.0001", "final_loss": loss, "test_acc": acc})

print("\n--- Experiment Results ---")
for r in results:
    print(r)