import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def extract_scalars(log_dir, scalar_name):
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()
    scalar_events = event_acc.Scalars(scalar_name)
    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    return steps, values

def find_best_run(log_base_dir):
    min_val_loss = float('inf')
    best_run_dir = None

    for run in os.listdir(log_base_dir):
        run_dir = os.path.join(log_base_dir, run)
        if os.path.isdir(run_dir):
            try:
                _, val_loss = extract_scalars(run_dir, 'val_loss')
                if val_loss and val_loss[-1] < min_val_loss:  # Compare final validation loss
                    min_val_loss = val_loss[-1]
                    best_run_dir = run_dir
            except Exception as e:
                print(f"Failed to process {run_dir}: {e}")

    print(f"Best run: {best_run_dir} with final val_loss: {min_val_loss}")
    return best_run_dir

def plot_best_run(log_base_dir):
    best_run_dir = find_best_run(log_base_dir)

    if best_run_dir:
        train_steps, train_values = extract_scalars(best_run_dir, 'train_loss')
        val_steps, val_values = extract_scalars(best_run_dir, 'val_loss')
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_values, label="Train Loss", linewidth=2)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves (Best Run)")
        plt.legend()
        plt.grid()
        plt.savefig("best_run_loss_curve_train.png")
        plt.clf()
        plt.plot(val_steps, val_values, label="Validation Loss", linewidth=2)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves (Best Run)")
        plt.legend()
        plt.grid()
        plt.savefig("best_run_loss_curve_val.png")
    else:
        print("No valid runs found!")

if __name__ == "__main__":
    log_base_dir = "tb_logs/optuna_run_augmented_new_2/"  
    plot_best_run(log_base_dir)
