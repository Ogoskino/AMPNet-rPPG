import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import os
import mlflow




# Bland-Altman plot function
def bland_altman_plot(hr_true, hr_pred, model_name, save_dir='./plots/BAPs'):

    os.makedirs(save_dir, exist_ok=True)
    mean_hr = np.mean([hr_true, hr_pred], axis=0)
    diff_hr = hr_true - hr_pred
    mean_diff = np.mean(diff_hr)
    std_diff = np.std(diff_hr)

    plt.figure(figsize=(10, 8))
    plt.scatter(mean_hr, diff_hr, color='red', alpha=0.5)
    plt.axhline(mean_diff, color='teal', linestyle='--', label=f'mean diff: {mean_diff:.2f}')
    plt.axhline(mean_diff + 1.96*std_diff, color='black', linestyle='--', label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    plt.axhline(mean_diff - 1.96*std_diff, color='black', linestyle='--', label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
    plt.xlabel(f'Mean of HR(True) and HR({model_name})', fontsize=14, fontweight='bold')
    plt.ylabel(f'(HR True - HR {model_name})', fontsize=14, fontweight='bold')
    plt.title(f'{model_name}', fontsize=14, fontweight='bold')
    #plt.legend(fontsize=12, frameon=True)
    plt.legend(prop={'size': 14, 'weight': 'bold'})
    #plt.grid(True)
    # Save the plot in the specified folder
    plot_filename = os.path.join(save_dir, f'hr_true_vs_pred_{model_name}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    # Log the plot as an artifact using MLflow
    mlflow.log_artifact(plot_filename)

# Scatter plot function for HR(true) vs HR(pred)
def scatter_plot(hr_true, hr_pred, model_name, save_dir='./plots/SPs'):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(hr_true, hr_pred, color='red', alpha=0.5)
    pcc, _ = pearsonr(hr_true, hr_pred)
    plt.plot([min(hr_true), max(hr_true)], [min(hr_true), max(hr_true)], color='green', linestyle='--', label=f'Identity line\n$\\mathbf{{r}}$: {pcc:.2f}')
    plt.xlabel('HR(True)', fontsize=20, fontweight='bold')
    plt.ylabel(f'HR({model_name})', fontsize=20, fontweight='bold')
    plt.title(f'{model_name}', fontsize=20, fontweight='bold')
    #plt.legend(fontsize=12, frameon=True)
    plt.legend(prop={'size': 20, 'weight': 'bold'})
    #plt.grid(True)
    # Save the plot in the specified folder
    plot_filename = os.path.join(save_dir, f'hr_true_vs_pred_{model_name}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    # Log the plot as an artifact using MLflow
    mlflow.log_artifact(plot_filename)


def plot_heart_rate(hr_true, hr_pred, model_name, save_dir='./plots/HRs'):
    """
    This function plots the ground truth vs predicted heart rate and logs it as an artifact.

    Args:
    - hr_true (numpy.ndarray): Ground truth heart rate values.
    - hr_pred (numpy.ndarray): Predicted heart rate values.
    - model_name (str): The name of the model being used.
    - save_dir (str): Directory to save the plots. Default is current directory.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plotting the heart rate
    plt.figure(figsize=(10, 8))
    plt.plot(hr_true, label='Ground-Truth', color='black', marker='o', markersize=3)
    plt.plot(hr_pred, label=model_name, color='teal', linestyle='--', marker='o', markersize=3)
    #plt.xlabel('Time')
    #plt.ylabel('Heart Rate in BPM')
    #plt.legend()

    plt.xlabel('Time', fontsize=20, fontweight='bold')
    plt.ylabel('Heart Rate in BPM', fontsize=20, fontweight='bold')
    plt.title(f'{model_name}', fontsize=20, fontweight='bold')
    #plt.legend(fontsize=12, frameon=True)
    plt.legend(prop={'size': 20, 'weight': 'bold'})

    # Save the plot in the specified folder
    plot_filename = os.path.join(save_dir, f'hr_true_vs_pred_{model_name}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    # Log the plot as an artifact using MLflow
    mlflow.log_artifact(plot_filename)



def plot_bvp_signals(fold_labels, fold_outputs, model_name, macc=None, timelag=None,
                     save_dir='./plots/bvps', fs=None):
    """
    Plots the first two BVP signals and writes MACC + timelag onto each plot.

    Args:
        fold_labels (list/array): true BVP signals per fold
        fold_outputs (list/array): predicted BVP signals per fold
        model_name (str): model name for legend/title
        macc (float or sequence, optional): MACC value(s). If sequence, one per plotted entry.
        timelag (int/float or sequence, optional): lag in samples (or in same units you pass).
        save_dir (str): directory to save plots
        fs (float, optional): sampling rate (Hz). If provided and timelag is in samples,
                              the lag in seconds will also be shown.
    """
    os.makedirs(save_dir, exist_ok=True)

    # helper to extract scalar or sequence element
    def _get_val(val, idx):
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            if idx < len(val):
                return val[idx]
            return val[0]  # fallback
        return val

    for i in range(2):
        y_true = np.asarray(fold_labels[i])
        y_pred = np.asarray(fold_outputs[i])

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(y_true, label='Ground-Truth', color='black')
        ax.plot(y_pred, label=model_name, color='teal', linestyle='--')
        ax.set_xlabel('Frames', fontsize=14, fontweight='bold')
        ax.set_ylabel('BVP Signal Amplitude', fontsize=14, fontweight='bold')
        ax.set_title(f'{model_name}_seq_{i+1}', fontsize=14, fontweight='bold')
        #ax.legend(loc='best', prop={'size': 12, 'weight': 'bold'})
        ax.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'})


        # get values for this plot
        m_val = _get_val(macc, i)
        lag_val = _get_val(timelag, i)

        # prepare text to display
        info_lines = []
        if m_val is not None:
            try:
                info_lines.append(f"MACC: {float(m_val):.3f}")
            except Exception:
                info_lines.append(f"MACC: {m_val}")
        if lag_val is not None:
            try:
                lag_s = float(lag_val)
                info_lines.append(f"Lag: {lag_s:.3f} secs")
            except Exception:
                info_lines.append(f"Lag: {lag_val}")

        if info_lines:
            info_text = "\n".join(info_lines)
            # move info-text to top-left instead
            ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
                    ha='left', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))


        plot_filename = os.path.join(save_dir, f'pred_true_bvp_{i+1}_{model_name}.png')
        fig.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)

        # log artifact
        try:
            mlflow.log_artifact(plot_filename)
        except Exception as e:
            # optional: print a warning if mlflow not configured
            print(f"Warning: failed to log artifact to mlflow: {e}")


# Example usage:
# Assuming you are in an active MLflow run
# plot_heart_rate(hr_true, hr_pred, 'R-3EDSAN', './plots')
# plot_bvp_signals(fold_labels, fold_outputs, 'AMPNet', './plots')
