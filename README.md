# Predicting Chaos: Machine Learning Classification of Duffing Oscillator Behaviors

A machine learning project that predicts chaotic vs. periodic behavior in the forced, damped Duffing oscillator using only system parameters—eliminating the need for expensive numerical integration during prediction.

**Author:** Ryan Hardig  
**Course:** Physics 5680, Autumn 2025  
**Date:** October 2025

## Overview

Chaos theory has long fascinated physicists and mathematicians due to its paradoxical nature: deterministic systems that produce seemingly unpredictable behavior. While traditional methods like computing Lyapunov exponents can identify chaos, they require solving differential equations repeatedly—a computationally expensive process.

This project demonstrates how **machine learning can serve as a fast surrogate** for chaos detection. By training models on a synthetically generated dataset of Duffing oscillator simulations, we learn to predict whether a given set of physical parameters will produce chaotic or periodic motion.

### The Duffing Oscillator

The forced and damped Duffing oscillator is governed by:

$$\ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t)$$

where:
- $\alpha$ and $\beta$ describe the stiffness of the potential
- $\delta$ is the damping coefficient
- $\gamma$ is the amplitude of an external driving force
- $\omega$ is the driving frequency

## Key Features

### Project Goals

1. **Binary Classification**: Predict whether system parameters lead to chaotic or periodic behavior
2. **Regression (Stretch Goal)**: Estimate the Largest Lyapunov Exponent (LLE) value from parameters alone

### Dataset

- **Size**: ~127,000 samples
- **Features**: 5 system parameters (α, β, δ, γ, ω)
- **Labels**: Binary (periodic/chaotic) based on Largest Lyapunov Exponent
- **Generation**: GPU-accelerated numerical integration using PyTorch
- **Class Distribution**: 59% periodic, 41% chaotic

### Models Implemented

- Random Forest Classifier (primary)
- Multi-Layer Perceptron (neural network)
- Support Vector Machines (SVM)
- Decision Trees (baseline)

## Project Structure

```
duffing/
├── solver.py              # Duffing equation numerical solver
├── generate_data.py       # Dataset generation with GPU support
├── features.py            # Feature extraction & LLE computation
├── model.py               # ML model training utilities
├── bifurcation_predict.py # Bifurcation analysis tools
├── visualize.py           # Plotting utilities
└── __init__.py

notebooks/
├── 01_generate_data.ipynb       # Dataset generation workflow
├── 02_chaos_analysis.ipynb      # Data exploration & analysis
├── 03_bifurcation_walkthrough.ipynb # Bifurcation diagrams
└── presentation.ipynb           # Final presentation

scripts/
├── run_batch_gpu_datasets.ps1   # GPU batch processing
├── run_train_on_csv.py          # Model training pipeline
└── check_chaos_counts.py        # Data validation

5680_final_project_Ryan_Hardig_2025.ipynb  # Complete final report
```

## Installation

### Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PyTorch (for GPU-accelerated ODE solving)
- Joblib

### Setup

```bash
# Clone the repository
git clone https://github.com/ryanhardig/ML-Duffing.git
cd ML-Duffing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Training Data

```python
from duffing.generate_data import generate_dataset

# Generate dataset with Lyapunov exponent computation
df = generate_dataset(
    n_param_sets=10000,
    out_csv='duffing_dataset.csv',
    compute_lyapunov=True,
    rng_seed=42
)
```

### 2. Train a Model

```python
from duffing.model import train_rf_model, load_dataset

# Load data
df = load_dataset('duffing_dataset.csv')

# Train Random Forest classifier
model, accuracy, precision, recall = train_rf_model(df, target='periodic')

# Save trained model
import joblib
joblib.dump(model, 'rf_model_periodic.joblib')
```

### 3. Make Predictions

```python
import joblib
import numpy as np

# Load trained model
model = joblib.load('rf_model_periodic.joblib')

# Predict on new parameters
# Format: [delta, alpha, beta, gamma, omega]
parameters = np.array([[0.25, 0.5, 1.0, 0.3, 0.8]])
prediction = model.predict(parameters)
probability = model.predict_proba(parameters)

print(f"Chaotic: {not prediction[0]}")
print(f"Confidence: {max(probability[0])*100:.1f}%")
```

### 4. Run Complete Workflow in Notebook

See `5680_final_project_Ryan_Hardig_2025.ipynb` for the full analysis pipeline including:
- Data loading and preprocessing
- Exploratory data analysis
- Model training and evaluation
- Performance metrics and visualizations
- Parameter sensitivity analysis

## Key Results

### Model Performance (Classification)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~95% | ~93% | ~96% | ~95% |
| MLP Neural Network | ~94% | ~92% | ~95% | ~94% |
| SVM | ~92% | ~90% | ~93% | ~92% |

### Computational Advantage

- **Traditional method**: ~5-10 seconds per parameter set (solving ODE + computing LLE)
- **ML prediction**: <1 millisecond per parameter set
- **Speedup**: **1000x faster** for exploring parameter space

## Scientific Background

### Chaos Detection via Lyapunov Exponents

The **Largest Lyapunov Exponent (LLE)** quantifies the exponential divergence of nearby trajectories:
- **LLE > 0**: Chaotic behavior
- **LLE < 0**: Periodic or quasi-periodic behavior
- **LLE ≈ 0**: Bifurcation boundary

We compute LLE using the **Benettin algorithm** (Wolf et al., 1985), which tracks perturbation growth over time.

### ML Surrogate Approach

Rather than computing LLE for every new parameter set, we train models that learn the mapping:

$$(α, β, δ, γ, ω) \rightarrow \{\text{chaotic, periodic}\}$$

This allows rapid parameter space exploration without repeated numerical integration.

## References

Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J.-M. (1980).
Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; A method for computing all of them. Part 1: Theory.
*Meccanica*, 15, 9–20.

Wolf, A., Swift, J. B., Swinney, H. L., & Vastano, J. A. (1985).
Determining Lyapunov exponents from a time series.
*Physica D: Nonlinear Phenomena*, 16(3), 285–317.

Pathak, J., Hunt, B. R., Girvan, M., Lu, Z., & Ott, E. (2017).
Using machine learning to replicate chaotic attractors and calculate Lyapunov exponents from data.
*Chaos: An Interdisciplinary Journal of Nonlinear Science*, 27(12), 121102.

Lee, G., Nelson, R., Hassona, S., et al. (2020).
Deep learning classification of chaos in the standard map.
*arXiv preprint*.

Hassona, S., Yao, Y., & Qureshi, M. A. (2021).
Time series classification and creation of 2D bifurcation diagrams using machine learning.
*Applied Soft Computing*, 108, 107448.

## Ethical Considerations

This project uses synthetically generated data from numerical simulations, with no personal or sensitive information involved. However, the approach highlights important considerations for ML in scientific research:

- **Model Limitations**: ML models reflect biases and limitations of their training data
- **Domain Boundaries**: Predictions are only reliable within the parameter ranges used for training
- **Verification**: ML surrogates should complement, not replace, classical numerical verification
- **Transparency**: Clear documentation of model capabilities and limitations is essential for responsible use

## Future Extensions

- **Other Chaotic Systems**: Generalize approach to Lorenz, Rössler, and other systems
- **Regression LLE**: Predict actual Lyapunov exponent values (not just binary classification)
- **Bifurcation Prediction**: Identify parameter regions near bifurcation boundaries
- **Uncertainty Quantification**: Provide confidence intervals for predictions
- **Advanced Architectures**: Explore graph neural networks, attention mechanisms
- **Sensitivity Analysis**: Identify which parameters most strongly influence chaos

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## Contact

**Ryan Hardig**  
Ohio State University  
Department of Physics  
Email: hardig.5@osu.edu

---

*This project was completed as part of Physics 5680: Chaos and Nonlinear Dynamics, Autumn 2025.*
