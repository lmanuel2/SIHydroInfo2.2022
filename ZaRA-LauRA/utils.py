import matplotlib.pyplot as plt

def loss_decay_plot(num_epochs, train_loss, val_loss, save_path):

    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), train_loss, color='g', label="Train")
    ax.plot(range(num_epochs), val_loss, color='r', label="Val")

    ax.set_xticks(range(0, num_epochs, 5))

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.title(f'Loss Decay')

    ax.grid(True)
    ax.legend()

    plt.savefig(f"{save_path}/loss_decay.png")
    plt.show()


def vs_q_plot(discharge, width, prediction, save_path):

    fig, ax = plt.subplots()
    ax.scatter(discharge, width, color='g', marker='.', label="Observations")
    ax.scatter(discharge, prediction, color='r', marker='.', label="Model")

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Discharge')
    ax.set_ylabel('Channel Width')

    ax.legend()
    ax.grid(True)

    plt.savefig(f"{save_path}/vs_q.png")
    plt.show()


def pred_vs_gt_plot(prediction, width, metric_dict, save_path):

    fig, ax = plt.subplots()
    ax.scatter(width, prediction, color='b', marker='.')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.grid(True)
    ax.set_xlabel('GT Width')
    ax.set_ylabel('Pred Width')
    plt.title(fr'NSE={metric_dict["NSE_log"]:.4f}, $R^2$: {metric_dict["r_value_log"]**2:.4f}, PBias: {metric_dict["pbias_log"]:.4f}')

    plt.savefig(f"{save_path}/pred_vs_gt.png")
    plt.show()