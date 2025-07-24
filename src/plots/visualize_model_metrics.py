import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import datetime

import pandas as pd

def plot(path):

    forecast_issuance_times = np.load(os.path.join(path, 'forecast_issuance_times.npy'), allow_pickle=True)
    real_ev = np.load(os.path.join(path, 'real_ev.npy'))
    real_energy = np.load(os.path.join(path, 'real_energy.npy'))
    forecast_ev = np.load(os.path.join(path, 'forecast_ev.npy'))
    forecast_energy = np.load(os.path.join(path, 'forecast_energy.npy'))


    # print(f"Real Energy: {type(real_ev)}-{real_ev.shape}")
    # print(f"Real Energy: {type(real_energy)}-{real_energy.shape}")
    # print(f"Forecast Issuance Times: {forecast_issuance_times}")
    # print(f"Forecast Energy: {type(forecast_ev)}-{forecast_ev.shape}")
    # print(f"Forecast Energy: {type(forecast_energy)}-{forecast_energy.shape}")

    # Compute mean absolute error
    mae_ev = np.abs(forecast_ev - real_ev[:, None, :]).mean(axis=0)  # (3, 96)
    mae_energy = np.abs(forecast_energy - real_energy[:, None, :]).mean(axis=0)  # (3, 96)

    # Compute the average real EV count and total energy across all days
    avg_real_ev = real_ev.mean(axis=0)
    avg_real_energy = real_energy.mean(axis=0)

    # Create time labels in HH:MM format (15-minute intervals)
    time_labels = [(datetime.datetime(2025, 1, 1, 0, 0) + datetime.timedelta(minutes=15 * i)).strftime("%H:%M") for i in range(96)]
    tick_positions = list(range(0, 96, 8))  

    time_labels_forecast_issuance_times = [forecast_issuance_times[i].strftime("%H:%M") for i in range(len(forecast_issuance_times))]
    tick_positions_forecast_issuance_times = list(range(0, len(forecast_issuance_times), 4))


    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    # Row 1: Real average EV and Energy
    axes[0, 0].plot(avg_real_ev, color='black')
    axes[0, 0].set_title("Average Real EV Count Over the Day")
    axes[0, 0].set_ylabel("EV Count")
    axes[0, 0].set_xticks(tick_positions)
    axes[0, 0].set_xticklabels([time_labels[i] for i in tick_positions], rotation=45)

    axes[0, 1].plot(avg_real_energy, color='black')
    axes[0, 1].set_title("Average Real Energy Over the Day")
    axes[0, 1].set_ylabel("Energy")
    axes[0, 1].set_xticks(tick_positions)
    axes[0, 1].set_xticklabels([time_labels[i] for i in tick_positions], rotation=45)


    # Heatmap - EV count
    sns.heatmap(mae_ev, cmap='magma', ax=axes[1, 0], xticklabels=12)
    axes[1, 0].set_title("MAE: Forecast vs Real EV Count")
    axes[1, 0].set_xlabel("Time (HH:MM)")
    axes[1, 0].set_ylabel("Issuance Time")
    axes[1, 0].set_xticks(tick_positions)
    axes[1, 0].set_xticklabels([time_labels[i] for i in tick_positions])
    axes[1, 0].set_yticks(tick_positions_forecast_issuance_times)
    axes[1, 0].set_yticklabels([time_labels_forecast_issuance_times[i] for i in tick_positions_forecast_issuance_times], rotation=0)

    # Heatmap - Energy
    sns.heatmap(mae_energy, cmap='viridis', ax=axes[1, 1], xticklabels=12,
                yticklabels=[str(t) for t in forecast_issuance_times])
    axes[1, 1].set_title("MAE: Forecast vs Real Energy")
    axes[1, 1].set_xlabel("Time (HH:MM)")
    axes[1, 1].set_ylabel("Issuance Time")
    axes[1, 1].set_xticks(tick_positions)
    axes[1, 1].set_xticklabels([time_labels[i] for i in tick_positions])
    axes[1, 1].set_yticks(tick_positions_forecast_issuance_times)
    axes[1, 1].set_yticklabels([time_labels_forecast_issuance_times[i] for i in tick_positions_forecast_issuance_times], rotation=0)

    # Line plot - EV count
    for i, t in enumerate(forecast_issuance_times):
        axes[2, 0].plot(mae_ev[i], label=f"Forecast @ {t}")
    axes[2, 0].set_title("EV Count MAE Over Forecast Horizon")
    axes[2, 0].set_xlabel("Time (HH:MM)")
    axes[2, 0].set_ylabel("MAE")
    axes[2, 0].grid(True)
    axes[2, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    axes[2, 0].set_xticks(tick_positions)
    axes[2, 0].set_xticklabels([time_labels[i] for i in tick_positions])

    # Line plot - Energy
    for i, t in enumerate(forecast_issuance_times):
        axes[2, 1].plot(mae_energy[i], label=f"Forecast @ {t}")
    axes[2, 1].set_title("Energy MAE Over Forecast Horizon")
    axes[2, 1].set_xlabel("Time (HH:MM)")
    axes[2, 1].set_ylabel("MAE")
    axes[2, 1].grid(True)
    axes[2, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    axes[2, 1].set_xticks(tick_positions)
    axes[2, 1].set_xticklabels([time_labels[i] for i in tick_positions])

    plt.tight_layout()
    plt.show()


def load_method_results(path):
    forecast_issuance_times = np.load(os.path.join(path, 'forecast_issuance_times.npy'), allow_pickle=True)
    real_ev = np.load(os.path.join(path, 'real_ev.npy'))
    real_energy = np.load(os.path.join(path, 'real_energy.npy'))
    forecast_ev = np.load(os.path.join(path, 'forecast_ev.npy'))
    forecast_energy = np.load(os.path.join(path, 'forecast_energy.npy'))

    mae_ev = np.abs(forecast_ev - real_ev[:, None, :]).mean(axis=0)
    mae_energy = np.abs(forecast_energy - real_energy[:, None, :]).mean(axis=0)
    avg_real_ev = real_ev.mean(axis=0)
    avg_real_energy = real_energy.mean(axis=0)

    return forecast_issuance_times, mae_ev, mae_energy, avg_real_ev, avg_real_energy

def compare_methods_compact(path_a, label_a, path_b, label_b):
    (times, mae_ev_a, mae_energy_a, avg_ev, avg_energy) = load_method_results(path_a)
    (_,    mae_ev_b, mae_energy_b, _,        _)         = load_method_results(path_b)

    rows = []
    for i, t in enumerate(times):
        rows.append({
            "Forecast Time": t,
            f"MAE EV ({label_a})": mae_ev_a[i].mean(),
            f"MAE EV ({label_b})": mae_ev_b[i].mean(),
            f"MAE Energy ({label_a})": mae_energy_a[i].mean(),
            f"MAE Energy ({label_b})": mae_energy_b[i].mean(),
            "Avg Real EV": avg_ev.mean(),
            "Avg Real Energy": avg_energy.mean(),
        })

    return pd.DataFrame(rows).round(2)


def plot_mae_vs_time(times, mae_ev_a, mae_ev_b, label_a='kNN', label_b='Bayes'):
    avg_mae_a = mae_ev_a.mean(axis=1)
    avg_mae_b = mae_ev_b.mean(axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(times, avg_mae_a, label=f'MAE EV ({label_a})', marker='o')
    plt.plot(times, avg_mae_b, label=f'MAE EV ({label_b})', marker='s')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Absolute Error (EV Count)')
    plt.xlabel('Forecast Issuance Time')
    plt.title('MAE of EV Forecasts Across Forecast Times')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mae_next_hour(times, mae_ev_a, mae_ev_b, label_a='kNN', label_b='Bayes', bin_width_mins=15, duration_mins=60):
    # Convert forecast times to strings for the x-axis
    times = [str(t) for t in times]

    n_bins = duration_mins // bin_width_mins  # 60 min => 4 bins

    avg_mae_a = mae_ev_a[:, :n_bins].mean(axis=1)
    avg_mae_b = mae_ev_b[:, :n_bins].mean(axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(times, avg_mae_a, label=f'Next 1h MAE ({label_a})', marker='o')
    plt.plot(times, avg_mae_b, label=f'Next 1h MAE ({label_b})', marker='s')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Absolute Error (EV Count)')
    plt.xlabel('Forecast Issuance Time')
    plt.title('MAE of EV Forecasts (Next 1 Hour)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()