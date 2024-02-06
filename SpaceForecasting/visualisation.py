import matplotlib.pyplot as plt


def plot_predictions(actual, predicted, title='Solar Wind Speed Forecast'):
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, label='Actual', alpha=0.7, marker='.', linestyle='-', linewidth=1.0)
    plt.plot(actual.index, predicted, label='Predicted', alpha=0.7, marker='x', linestyle='--', linewidth=1.0)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Solar Wind Speed')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
