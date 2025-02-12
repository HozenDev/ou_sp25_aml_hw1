import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_bmi_dataset(filepath):
    """
    Load the BMI dataset from a pickle file.
    :param filepath: Path to the pickle file
    :return: Dictionary containing dataset
    """
    with open(filepath, "rb") as fp:
        data = pickle.load(fp)
    return data


def summarize_dataset(data):
    """
    Print summary statistics for each key in the dataset.
    :param data: Dictionary containing dataset
    """
    for key, value in data.items():
        print(f"{key}:")
        print(f"  Number of folds: {len(value)}")
        print(f"  Shape of first fold: {value[0].shape if isinstance(value[0], np.ndarray) else 'Unknown'}")
        if isinstance(value[0], np.ndarray):
            combined_data = np.concatenate(value, axis=0)
            print(f"  Range: {combined_data.min()} to {combined_data.max()}")
            print(f"  Mean: {combined_data.mean()}")
            print(f"  Standard Deviation: {combined_data.std()}\n")


def plot_sample_time_series(data, fold=0):
    """
    Plot time series of selected features from a given fold.
    :param data: Dictionary containing dataset
    :param fold: Fold index to visualize
    """
    time = data["time"][fold]
    theta = data["theta"][fold]
    dtheta = data["dtheta"][fold]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, theta[:, 0], label="Shoulder Position")
    plt.plot(time, theta[:, 1], label="Elbow Position")
    plt.ylabel("Position")
    plt.legend()
    plt.title("Theta (Angular Position) Over Time")
    
    plt.subplot(2, 1, 2)
    plt.plot(time, dtheta[:, 0], label="Shoulder Velocity")
    plt.plot(time, dtheta[:, 1], label="Elbow Velocity")
    plt.xlabel("Time (ms)")
    plt.ylabel("Velocity")
    plt.legend()
    plt.title("dTheta (Angular Velocity) Over Time")
    
    plt.tight_layout()
    plt.show()


# dataset = load_bmi_dataset("/home/fagg/datasets/bmi/bmi_dataset.pkl")
# summarize_dataset(dataset)
# plot_sample_time_series(dataset, fold=0)
