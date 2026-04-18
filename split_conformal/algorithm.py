#!/usr/bin/env python3
"""
Algorithm Module

Contains implementations of two online conformal inference algorithms:
  1. DriftDetectionConformal (DriftOCP) — Our proposed method with drift detection
  2. QuantileAdaptiveCI (ACI)           — Baseline method with 5 step-size variants

Core idea comparison:
  - ACI:      Uses fixed/decaying step sizes to update quantile online; does not actively detect drift
  - DriftOCP: Monitors cumulative sum of coverage errors; resets quantile when drift is detected
  
Both methods output a quantile sequence q_1, q_2, ..., q_T,
used to construct prediction intervals [Y_hat_t - q_t, Y_hat_t + q_t].
"""

import numpy as np                                           # Core numerical computation library

from drift_detection_conformal import DriftDetectionConformal  # DriftOCP algorithm class from the same directory


# ==================== ACI Baseline Algorithm ====================
class QuantileAdaptiveCI:
    """
    Quantile-based Adaptive Conformal Inference (ACI)
    
    This is the baseline algorithm compared in the paper.
    
    Algorithm principle:
      Maintains a quantile threshold q_t online, updated after each new observation:
        s_t = |Y_t - Y_hat_t|                          (conformal score)
        q_{t+1} = q_t + eta_t * (1{s_t > q_t} - alpha) (online update)
      
      Intuition:
        - If s_t > q_t (prediction interval fails to cover the true value), indicator=1 > alpha, q increases (interval widens)
        - If s_t <= q_t (covered), indicator=0 < alpha, q decreases (interval narrows)
        - In the long run, the miscoverage rate converges to alpha
    
    Step-size variants:
      - decaying_0.6: eta_t = (t+n+1)^{-0.6}  — Faster decay, quick convergence but weaker adaptivity
      - decaying_0.5: eta_t = (t+n+1)^{-0.5}  — Moderate decay, theoretically optimal (for static distributions)
      - fixed 0.01:   eta_t = 0.01             — Small fixed step, conservative but stable
      - fixed 0.1:    eta_t = 0.1              — Medium fixed step, balances adaptivity and stability
      - fixed 0.5:    eta_t = 0.5              — Large fixed step, fast adaptation but high oscillation
    """
    
    def __init__(self, model, X_train, Y_train, X_predict, Y_predict):
        """
        Initialize the ACI algorithm
        
        Parameters:
            model:     sklearn prediction model (e.g., RandomForestRegressor)
            X_train:   Training features, shape=(n_train, n_features), used to fit the model
            Y_train:   Training labels, shape=(n_train,), used to fit the model
            X_predict: Test features, shape=(T, n_features), arriving online step by step
            Y_predict: Test true labels, shape=(T,), revealed online step by step
        """
        self.model = model                                   # Store prediction model reference
        self.X_train = X_train                               # Store training features
        self.Y_train = Y_train                               # Store training labels
        self.X_predict = X_predict                           # Store test features (online data stream)
        self.Y_predict = Y_predict                           # Store test true labels
        self.model.fit(X_train, Y_train)                     # Fit model on training set (trained only once)
        self.train_predictions = self.model.predict(X_train) # Compute training predictions (used to initialize q_0)
        self.test_predictions = self.model.predict(X_predict)# Pre-compute test predictions (speed up subsequent computation)
    
    def compute_quantile_adaptive_intervals(self, alpha=0.1, step_type='decaying_0.6', fixed_eta=0.01):
        """
        Run the ACI algorithm and return the quantile threshold sequence at each step
        
        Parameters:
            alpha:     Target miscoverage rate, default 0.1 (i.e., expected 90% coverage)
            step_type: Step-size type
                       'decaying_0.6' — Decaying step eta_t = (t+n+1)^{-0.6}
                       'decaying_0.5' — Decaying step eta_t = (t+n+1)^{-0.5}
                       'fixed'        — Fixed step eta_t = fixed_eta
            fixed_eta: Step-size value used when step_type='fixed'
        
        Returns:
            quantiles: Quantile threshold at each step, shape=(T,)
                       q_t represents the prediction interval half-width output by the algorithm at step t
        """
        n = len(self.X_train)                                # Number of training samples (used in decaying step formula)
        T = len(self.X_predict)                              # Total number of test time steps
        
        # ---- Initialize q_0: (1-alpha) quantile of training residuals ----
        train_residuals = np.abs(self.Y_train - self.train_predictions)  # Training residuals: |Y - Y_hat|
        q_t = np.quantile(train_residuals, 1 - alpha)        # Take (1-alpha) quantile, e.g., 90th percentile when alpha=0.1
        
        quantiles = []                                       # Store quantile value at each step

        # ---- Online update loop ----
        for t in range(T):                                   # Process step by step from t=0 to t=T-1
            # Compute conformal score at step t
            s_t = np.abs(self.Y_predict[t] - self.test_predictions[t])  # s_t = |Y_t - Y_hat_t|
            
            quantiles.append(q_t)                            # Record current q_t first (determined before observing s_t)
            
            # Compute step size eta_t
            if step_type == 'decaying_0.6':                  # Decaying step, exponent 0.6
                eta_t = (t + n + 1) ** (-0.6)                # eta_t = (t+n+1)^{-0.6}
            elif step_type == 'decaying_0.5':                # Decaying step, exponent 0.5
                eta_t = (t + n + 1) ** (-0.5)                # eta_t = (t+n+1)^{-0.5}
            elif step_type == 'fixed':                       # Fixed step size
                eta_t = fixed_eta                            # eta_t = constant
            else:                                            # Unknown type, raise error
                raise ValueError(f"Unknown step_type: {step_type}")
            
            # ACI core update formula
            indicator = 1.0 if s_t > q_t else 0.0           # 1{s_t > q_t}: equals 1 if score exceeds threshold
            q_t = q_t + eta_t * (indicator - alpha)          # q_{t+1} = q_t + eta_t * (1{s>q} - alpha)
            # Meaning: if miscovered (indicator=1), then (1-alpha)>0, q increases; if covered (indicator=0), then (-alpha)<0, q decreases
            
            q_t = max(q_t, 0)                                # Ensure q is non-negative (interval width cannot be negative)
        
        return np.array(quantiles)                           # Convert to numpy array and return, shape=(T,)
