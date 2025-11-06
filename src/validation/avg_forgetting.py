import numpy as np

# Example accuracy matrix for 10 tasks
# Each row = after training a task
# Each column = test accuracy on each task
# (Here, values are just for demonstration)
accuracies = np.array([
    [0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.78, 0.82, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.70, 0.77, 0.83, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.68, 0.74, 0.80, 0.84, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.66, 0.73, 0.78, 0.82, 0.85, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.64, 0.70, 0.75, 0.80, 0.83, 0.86, 0.00, 0.00, 0.00, 0.00],
    [0.63, 0.69, 0.73, 0.78, 0.81, 0.85, 0.87, 0.00, 0.00, 0.00],
    [0.61, 0.67, 0.72, 0.76, 0.80, 0.83, 0.86, 0.88, 0.00, 0.00],
    [0.59, 0.66, 0.70, 0.75, 0.78, 0.82, 0.85, 0.87, 0.89, 0.00],
    [0.58, 0.64, 0.69, 0.74, 0.77, 0.81, 0.84, 0.86, 0.88, 0.90],
])

def average_accuracy(acc_matrix):
    """Compute the average accuracy after the last task."""
    final_accuracies = acc_matrix[-1]
    return np.mean(final_accuracies)

def average_forgetting(acc_matrix):
    """Compute the average forgetting over all tasks."""
    n_tasks = acc_matrix.shape[1]
    forgetting = []
    for t in range(n_tasks - 1):
        max_acc = np.max(acc_matrix[: , t])  # best accuracy on task t
        last_acc = acc_matrix[-1, t]         # accuracy on task t after all training
        forgetting.append(max_acc - last_acc)
    return np.mean(forgetting)

# Compute metrics
avg_acc = average_accuracy(accuracies)
avg_fgt = average_forgetting(accuracies)

# Print results
print(f"Average Accuracy: {avg_acc:.4f}")
print(f"Average Forgetting: {avg_fgt:.4f}")