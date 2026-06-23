"""
Fast SVR Training Module
Fitness: RMSE (denormalized ke Ton)
- pso_training_direct(): Holdout mode - evaluasi langsung di test set
- pso_training_cv(): CV mode - K-Fold CV di training set
- validate_cv_fast(): Validasi parameter tunggal di N fold
- train_final_model_fast(): Final train + test
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time


def evaluate_direct(params, X_train, y_train, X_test, y_test, scaler_y, timeout=20):
    """Evaluasi SVR langsung: train di X_train, predict di X_test, RMSE denormalized"""
    C, epsilon, gamma = params
    try:
        model = SVR(kernel='rbf', C=float(C), epsilon=float(epsilon), gamma=float(gamma), max_iter=10000)
        model.fit(X_train, y_train)
        y_pred_scaled = model.predict(X_test)

        y_pred_asli = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test_asli = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_asli = np.clip(y_pred_asli, 0, None)

        rmse = float(np.sqrt(mean_squared_error(y_test_asli, y_pred_asli)))
        return rmse, y_pred_scaled, model
    except Exception:
        return float('inf'), None, None


def pso_training_direct(X_train, y_train, X_test, y_test, scaler_y,
                        n_particles, n_iter,
                        c_min, c_max, eps_min, eps_max, gamma_min, gamma_max,
                        w=0.7, c1=1.5, c2=1.5):
    """PSO holdout: fitness = RMSE di test set"""
    np.random.seed(42)
    lb = np.array([c_min, eps_min, gamma_min], dtype=float)
    ub = np.array([c_max, eps_max, gamma_max], dtype=float)

    particles = np.random.uniform(lb, ub, (n_particles, 3))
    velocities = np.zeros((n_particles, 3))
    personal_best = particles.copy()
    personal_best_score = np.array([float('inf')] * n_particles)

    global_best = None
    global_best_score = float('inf')
    global_best_pred = None
    global_best_model = None
    rmse_history = []

    X_train_a = np.asarray(X_train, dtype=np.float64)
    y_train_a = np.asarray(y_train, dtype=np.float64)
    X_test_a = np.asarray(X_test, dtype=np.float64)
    y_test_a = np.asarray(y_test, dtype=np.float64)

    for iter_num in range(n_iter):
        iter_info = {'iteration': int(iter_num + 1), 'total': int(n_iter), 'particles': []}

        for j in range(n_particles):
            C = float(np.clip(particles[j][0], c_min, c_max))
            eps = float(np.clip(particles[j][1], eps_min, eps_max))
            gamma = float(np.clip(particles[j][2], gamma_min, gamma_max))

            score, pred, model = evaluate_direct([C, eps, gamma], X_train_a, y_train_a, X_test_a, y_test_a, scaler_y)

            if score < personal_best_score[j]:
                personal_best[j] = particles[j].copy()
                personal_best_score[j] = score
                if score < global_best_score:
                    global_best_score = score
                    global_best = particles[j].copy()
                    global_best_pred = pred
                    global_best_model = model

            iter_info['particles'].append({
                'idx': int(j + 1),
                'rmse': float(score),
                'C': float(C),
                'eps': float(eps),
                'gamma': float(gamma)
            })

        rmse_history.append(float(global_best_score))
        iter_info['best_rmse'] = float(global_best_score)
        iter_info['progress'] = float((iter_num + 1) / n_iter)

        if global_best is not None:
            iter_info['best_params'] = {
                'C': float(global_best[0]),
                'epsilon': float(global_best[1]),
                'gamma': float(global_best[2])
            }

        if iter_num == n_iter - 1:
            iter_info['predictions'] = global_best_pred
            iter_info['model'] = global_best_model
            iter_info['rmse_history'] = rmse_history

        r1, r2 = float(np.random.rand()), float(np.random.rand())
        gb = global_best if global_best is not None else personal_best[0]
        for j in range(n_particles):
            velocities[j] = (
                w * velocities[j] +
                c1 * r1 * (personal_best[j] - particles[j]) +
                c2 * r2 * (gb - particles[j])
            )
            particles[j] += velocities[j]
            particles[j] = np.clip(particles[j], lb, ub)

        yield iter_info


def evaluate_cv(params, folds_data):
    """Evaluasi SVR dengan CV: rata-rata RMSE dari beberapa fold"""
    C, epsilon, gamma = params
    rmse_scores = []
    try:
        for X_tr, y_tr_sc, X_val, y_val_sc, scaler_fold, y_val_asli in folds_data:
            model = SVR(kernel='rbf', C=float(C), epsilon=float(epsilon), gamma=float(gamma), max_iter=10000)
            model.fit(X_tr, y_tr_sc)
            y_pred_sc = model.predict(X_val)
            y_pred_asli = scaler_fold.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
            y_pred_asli = np.clip(y_pred_asli, 0, None)
            rmse = float(np.sqrt(mean_squared_error(y_val_asli, y_pred_asli)))
            rmse_scores.append(rmse)
        return float(np.mean(rmse_scores))
    except Exception:
        return float('inf')


def pso_training_cv(X_train, y_train_scaled, scaler_y,
                    n_particles, n_iter,
                    c_min, c_max, eps_min, eps_max, gamma_min, gamma_max,
                    w=0.7, c1=1.5, c2=1.5, n_folds=5):
    """PSO dengan K-Fold CV: fitness = rata-rata RMSE antar fold"""
    np.random.seed(42)
    lb = np.array([c_min, eps_min, gamma_min], dtype=float)
    ub = np.array([c_max, eps_max, gamma_max], dtype=float)

    particles = np.random.uniform(lb, ub, (n_particles, 3))
    velocities = np.zeros((n_particles, 3))
    personal_best = particles.copy()
    personal_best_score = np.array([float('inf')] * n_particles)

    global_best = None
    global_best_score = float('inf')
    rmse_history = []

    X_train_a = np.asarray(X_train, dtype=np.float64)
    y_train_scaled_a = np.asarray(y_train_scaled, dtype=np.float64)

    kf = KFold(n_splits=n_folds, shuffle=False)
    folds_data = []
    y_train_asli = scaler_y.inverse_transform(y_train_scaled_a.reshape(-1, 1)).ravel()

    for train_idx, val_idx in kf.split(X_train_a):
        folds_data.append((
            X_train_a[train_idx],
            y_train_scaled_a[train_idx],
            X_train_a[val_idx],
            y_train_scaled_a[val_idx],
            scaler_y,
            y_train_asli[val_idx]
        ))

    for iter_num in range(n_iter):
        iter_info = {'iteration': int(iter_num + 1), 'total': int(n_iter), 'particles': []}

        for j in range(n_particles):
            C = float(np.clip(particles[j][0], c_min, c_max))
            eps = float(np.clip(particles[j][1], eps_min, eps_max))
            gamma = float(np.clip(particles[j][2], gamma_min, gamma_max))

            score = evaluate_cv([C, eps, gamma], folds_data)

            if score < personal_best_score[j]:
                personal_best[j] = particles[j].copy()
                personal_best_score[j] = score
                if score < global_best_score:
                    global_best_score = score
                    global_best = particles[j].copy()

            iter_info['particles'].append({
                'idx': int(j + 1),
                'rmse': float(score),
                'C': float(C),
                'eps': float(eps),
                'gamma': float(gamma)
            })

        rmse_history.append(float(global_best_score))
        iter_info['best_rmse'] = float(global_best_score)
        iter_info['progress'] = float((iter_num + 1) / n_iter)

        if global_best is not None:
            iter_info['best_params'] = {
                'C': float(global_best[0]),
                'epsilon': float(global_best[1]),
                'gamma': float(global_best[2])
            }

        if iter_num == n_iter - 1:
            iter_info['rmse_history'] = rmse_history

        r1, r2 = float(np.random.rand()), float(np.random.rand())
        gb = global_best if global_best is not None else personal_best[0]
        for j in range(n_particles):
            velocities[j] = (
                w * velocities[j] +
                c1 * r1 * (personal_best[j] - particles[j]) +
                c2 * r2 * (gb - particles[j])
            )
            particles[j] += velocities[j]
            particles[j] = np.clip(particles[j], lb, ub)

        yield iter_info


def validate_cv_fast(best_params, X_train, y_train_scaled, scaler_y, n_folds=10):
    """Validasi parameter terbaik dengan K-Fold CV"""
    C = float(best_params['C'])
    epsilon = float(best_params['epsilon'])
    gamma = float(best_params['gamma'])

    kf = KFold(n_splits=n_folds, shuffle=False)
    fold_results = []
    X_train_a = np.asarray(X_train, dtype=np.float64)
    y_train_scaled_a = np.asarray(y_train_scaled, dtype=np.float64)
    y_train_asli = scaler_y.inverse_transform(y_train_scaled_a.reshape(-1, 1)).ravel()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_a)):
        try:
            model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma, max_iter=10000)
            model.fit(X_train_a[train_idx], y_train_scaled_a[train_idx])
            y_pred_sc = model.predict(X_train_a[val_idx])
            y_pred_asli = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
            y_pred_asli = np.clip(y_pred_asli, 0, None)
            rmse = float(np.sqrt(mean_squared_error(y_train_asli[val_idx], y_pred_asli)))
        except Exception:
            rmse = float('inf')

        fold_results.append({
            'fold': int(fold_idx + 1),
            'rmse': rmse,
            'n_train': int(len(train_idx)),
            'n_val': int(len(val_idx))
        })

    valid_rmse = [r['rmse'] for r in fold_results if r['rmse'] != float('inf')]
    avg_rmse = float(np.mean(valid_rmse)) if valid_rmse else float('inf')
    return fold_results, avg_rmse


def train_final_model_fast(best_params, X_train, y_train_scaled, scaler_y, X_test, y_test_scaled):
    """Train final model di 100% training, predict test set"""
    C = float(best_params['C'])
    epsilon = float(best_params['epsilon'])
    gamma = float(best_params['gamma'])

    X_train_a = np.asarray(X_train, dtype=np.float64)
    y_train_a = np.asarray(y_train_scaled, dtype=np.float64)
    X_test_a = np.asarray(X_test, dtype=np.float64)
    y_test_a = np.asarray(y_test_scaled, dtype=np.float64)

    model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma, max_iter=10000)
    model.fit(X_train_a, y_train_a)

    y_pred_sc = model.predict(X_test_a)
    y_pred_asli = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    y_pred_asli = np.clip(y_pred_asli, 0, None)
    y_test_asli = scaler_y.inverse_transform(y_test_a.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_test_asli, y_pred_asli)))
    ss_res = np.sum((y_test_asli - y_pred_asli) ** 2)
    ss_tot = np.sum((y_test_asli - np.mean(y_test_asli)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return model, y_pred_asli, y_test_asli, rmse, r2
