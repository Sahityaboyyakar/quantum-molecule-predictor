import numpy as np
import pennylane as qml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def make_data(n_samples: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    # 4 sample chemical features (like size, solubility, polarity, bonds)
    X = rng.normal(0.0, 1.0, size=(n_samples, 4))
    y = (
        1.2 * np.sin(X[:, 0])
        + 0.7 * (X[:, 1] ** 2)
        - 0.8 * np.cos(X[:, 2] * X[:, 3])
        + rng.normal(0.0, 0.1, size=n_samples)
    )
    return X, y


def run_experiment():
    X, y = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Standard (classical) model
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classical_model = Ridge(alpha=1.0).fit(X_train_s, y_train)
    pred_classical = classical_model.predict(X_test_s)

    print("Classical RMSE:", round(float(np.sqrt(mean_squared_error(y_test, pred_classical))), 4))
    print("Classical R2:  ", round(float(r2_score(y_test, pred_classical)), 4))

    # 2. Quantum circuit feature model
    q_scaler = MinMaxScaler()
    X_train_q = q_scaler.fit_transform(X_train)
    X_test_q = q_scaler.transform(X_test)

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def quantum_circuit(features):
        for i in range(4):
            qml.RY(np.pi * features[i], wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    Z_train = np.array([quantum_circuit(row) for row in X_train_q])
    Z_test = np.array([quantum_circuit(row) for row in X_test_q])

    quantum_model = Ridge(alpha=0.5).fit(Z_train, y_train)
    pred_quantum = quantum_model.predict(Z_test)

    print("Quantum RMSE:  ", round(float(np.sqrt(mean_squared_error(y_test, pred_quantum))), 4))
    print("Quantum R2:    ", round(float(r2_score(y_test, pred_quantum)), 4))


if __name__ == "__main__":
    run_experiment()