"""
Fast SVR ANOVA RBF Training Module - Simplified version
Based on custom ANOVA RBF kernel with PSO optimization
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import time
from functools import partial

def calculate_mape(y_true, y_pred):
    """Calculate MAPE - sesuai dengan formula di Colab"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0 
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    mape_values = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered) * 100
    mape_values = np.clip(mape_values, 0, 100)  # Batasi maksimal 100% per data point
    return np.mean(mape_values)

def hitung_anova_rbf(X, Y, gamma, degree=2):
    """Custom ANOVA RBF kernel"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    K = np.zeros((X.shape[0], Y.shape[0]))
    
    for k in range(X.shape[1]):
        diff_sq = (X[:, k].reshape(-1, 1) - Y[:, k].reshape(1, -1)) ** 2
        K += np.exp(-gamma * diff_sq)
        
    # Dibagi jumlah fitur agar nilai matriks stabil (Normalized Kernel)
    K = K / X.shape[1] 
    
    return K ** degree

def evaluate_svr_anova(C, epsilon, gamma, X_train, y_train, X_test, y_test, timeout=8, max_iter=10000):
    """Quick SVR ANOVA RBF evaluation with timeout - sesuai dengan Colab setting"""
    start = time.time()
    try:
        # ✅ KONVERSI KE FLOAT64 UNTUK PRECISION SEPERTI COLAB (CRITICAL!)
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)
        
        # Ensure arrays are valid
        # Validate inputs
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            return float('inf'), None, None
        
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            return float('inf'), None, None
        
        # Use partial to pass gamma to kernel function
        kernel_func = partial(hitung_anova_rbf, gamma=gamma)
        model = SVR(kernel=kernel_func, C=C, epsilon=epsilon, max_iter=max_iter)
        model.fit(X_train, y_train)
        
        # Check timeout
        if time.time() - start > timeout:
            return float('inf'), None, None
        
        y_pred = model.predict(X_test)
        
        # Validate predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return float('inf'), None, None
        
        mape_score = calculate_mape(y_test, y_pred)
        
        # If MAPE is inf or nan, return inf
        if np.isnan(mape_score) or np.isinf(mape_score):
            return float('inf'), None, None
        
        return mape_score, y_pred, model
    except Exception as e:
        return float('inf'), None, None

def pso_training_anova(X_train, y_train, X_test, y_test, 
                       n_particles, n_iter, c_min, c_max, eps_min, eps_max, gamma_min, gamma_max,
                       w=0.7, c1=1.5, c2=1.5):
    """Fast PSO training with ANOVA RBF kernel - online progress - sesuai Colab (w=0.7)"""
    # Initialize
    np.random.seed(42)
    lb = np.array([c_min, eps_min, gamma_min])
    ub = np.array([c_max, eps_max, gamma_max])
    
    particles = np.random.uniform(lb, ub, (n_particles, 3))
    velocities = np.zeros((n_particles, 3))
    personal_best = particles.copy()
    personal_best_score = np.array([float('inf')] * n_particles)
    
    global_best = None
    global_best_score = float('inf')
    global_best_pred = None
    global_best_model = None
    mape_history = []
    
    # PSO Loop
    for iter_num in range(n_iter):
        iter_info = {'iteration': iter_num + 1, 'total': n_iter, 'particles': []}
        
        for j in range(n_particles):
            C, eps, gamma = particles[j]
            mape_score, y_pred, model = evaluate_svr_anova(
                C, eps, gamma, X_train, y_train, X_test, y_test,
                timeout=8, max_iter=10000
            )
            
            if mape_score < personal_best_score[j]:
                personal_best[j] = particles[j].copy()
                personal_best_score[j] = mape_score
                
                if mape_score < global_best_score:
                    global_best_score = mape_score
                    global_best = particles[j].copy()
                    global_best_pred = y_pred
                    global_best_model = model
            
            iter_info['particles'].append({
                'idx': j + 1,
                'mape': mape_score,
                'C': C,
                'eps': eps,
                'gamma': gamma
            })
        
        mape_history.append(global_best_score)
        iter_info['best_mape'] = global_best_score
        iter_info['progress'] = (iter_num + 1) / n_iter
        
        # Always include best_params for safe access
        if global_best is not None:
            iter_info['best_params'] = {
                'C': float(global_best[0]),
                'epsilon': float(global_best[1]),
                'gamma': float(global_best[2])
            }
        
        # Include final data on last iteration
        if iter_num == n_iter - 1:
            iter_info['predictions'] = global_best_pred
            iter_info['model'] = global_best_model
            iter_info['mape_history'] = mape_history
        
        # Update velocity and position
        r1, r2 = np.random.rand(), np.random.rand()
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
