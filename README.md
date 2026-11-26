@"
# Quantum Molecule Predictor

Compares a classical Ridge regression baseline against a 4-qubit PennyLane quantum circuit on synthetic molecular descriptors.

## Results
- Classical Ridge RMSE: 0.858 | R2: 0.5327
- Quantum Feature Ridge RMSE: 0.8147 | R2: 0.5786

## Run
~~~bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy scikit-learn pennylane
python run.py
~~~
"@ | Out-File -FilePath README.md -Encoding utf8