"""High-level runner to generate dataset, train model, and visualize a sample."""
from duffing.generate_data import generate_dataset
from duffing.model import train_model, load_dataset
from duffing.solver import solve_duffing
from duffing.visualize import plot_poincare, semilog_delta_x
import matplotlib.pyplot as plt


def main():
    csv = 'duffing_dataset.csv'
    print('Generating data...')
    df = generate_dataset(20, csv)
    print('Training model...')
    model, stats = train_model(load_dataset(csv))
    print('MSE:', stats['mse'])

    # visualize first sample
    row = df.iloc[0]
    params = {k: row[k] for k in ['delta', 'alpha', 'beta', 'gamma', 'omega']}
    sol = solve_duffing((0, 200), (0.1, 0.0), params)
    x = sol.y[0]
    v = sol.y[1]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    plot_poincare(sol.t, x, v, params['omega'], ax=axs[0])
    semilog_delta_x(sol.t, x, ax=axs[1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
