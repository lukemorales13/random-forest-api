from typing import Optional, Union
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder

class SimpleRandomForest:
    """
    Random Forest minimalista (bagging de árboles) con interfaz estilo scikit-learn.

    Parámetros
    ----------
    n_estimators : int, default=100
    max_features : {'sqrt','log2','all'} o int o float en (0,1], default='sqrt'
    criterion : str, default='gini' (clasif) o 'squared_error' (regresión)
    max_depth : int o None
    random_state : int o None
    task : {'auto','classification','regression'}
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Union[str, int, float] = "sqrt",
        criterion: Optional[str] = None,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        task: str = "auto",
    ):
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.task = task

        self.estimators_ = []
        self.task_ = None
        self.classes_ = None
        self._le = None
        self.n_features_in_ = None
        self._rng = None

    # -------- utilidades internas --------
    def _resolve_max_features(self, n_features: int):
        mf = self.max_features
        if mf == "all":
            return None  # sklearn usa None para "todas"
        if isinstance(mf, str):
            if mf in {"sqrt", "log2"}:
                return mf
            raise ValueError("max_features string debe ser 'sqrt', 'log2' o 'all'.")
        if isinstance(mf, (int, np.integer)):
            if 1 <= mf <= n_features:
                return int(mf)
            raise ValueError("max_features int debe estar en [1, n_features].")
        if isinstance(mf, float):
            if 0.0 < mf <= 1.0:
                return max(1, int(np.floor(mf * n_features)))
            raise ValueError("max_features float debe estar en (0, 1].")
        raise ValueError("max_features no válido.")

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState):
        n_samples = X.shape[0]
        idx = rng.randint(0, n_samples, size=n_samples)  # con reemplazo
        return X[idx], y[idx]

    def _infer_task(self, y: np.ndarray) -> str:
        if self.task in ("classification", "regression"):
            return self.task
        y_type = type_of_target(y)
        if y_type in ("binary", "multiclass"):
            return "classification"
        elif y_type in ("continuous",):
            return "regression"
        return "classification" if np.issubdtype(np.asarray(y).dtype, np.integer) else "regression"

    # -------------- API --------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X debe ser 2D (n_samples, n_features).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X y y deben tener el mismo número de filas.")

        self.n_features_in_ = X.shape[1]
        self.task_ = self._infer_task(y)
        self._rng = np.random.RandomState(self.random_state)
        max_features = self._resolve_max_features(self.n_features_in_)
        self.estimators_ = []

        if self.task_ == "classification":
            self._le = LabelEncoder().fit(y)
            self.classes_ = self._le.classes_
            criterion = self.criterion if self.criterion is not None else "gini"
            for _ in range(self.n_estimators):
                Xi, yi = self._bootstrap_sample(X, y, self._rng)
                seed = int(self._rng.randint(0, 2**31 - 1))
                tree = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=self.max_depth,
                    max_features=max_features,
                    random_state=seed,
                )
                tree.fit(Xi, yi)
                self.estimators_.append(tree)
        else:
            criterion = self.criterion if self.criterion is not None else "squared_error"
            for _ in range(self.n_estimators):
                Xi, yi = self._bootstrap_sample(X, y, self._rng)
                seed = int(self._rng.randint(0, 2**31 - 1))
                tree = DecisionTreeRegressor(
                    criterion=criterion,
                    max_depth=self.max_depth,
                    max_features=max_features,
                    random_state=seed,
                )
                tree.fit(Xi, yi)
                self.estimators_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("El modelo no está entrenado. Llama a fit(X, y) primero.")
        X = np.asarray(X)
        if self.task_ == "classification":
            proba = self.predict_proba(X)
            y_ind = np.argmax(proba, axis=1)
            return self.classes_[y_ind]
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        return preds.mean(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_ != "classification":
            raise AttributeError("predict_proba solo está disponible para clasificación.")
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n_samples, n_classes), dtype=float)
        for est in self.estimators_:
            est_proba = est.predict_proba(X)
            est_classes = est.classes_
            idx_map = np.searchsorted(self.classes_, est_classes)
            proba_sum[:, idx_map] += est_proba
        proba_avg = proba_sum / float(len(self.estimators_))
        row_sums = proba_avg.sum(axis=1, keepdims=True)
        zero_rows = (row_sums[:, 0] == 0.0)
        if np.any(zero_rows):
            proba_avg[zero_rows, :] = 1.0 / n_classes
            row_sums = proba_avg.sum(axis=1, keepdims=True)
        proba_avg /= row_sums
        return proba_avg

    def get_params(self, deep: bool = True):
        return {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "task": self.task,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

