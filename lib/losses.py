import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError
        
    def backward(self):
        raise NotImplementedError

class MSE(Loss):
    """
    Mean Square Error Loss
    """
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.N = y_pred.size
        self.error = y_pred - y_true
        
        # FIX: You wrote error**2 instead of self.error**2
        return (1 / self.N) * np.sum(self.error ** 2) 
        
    def backward(self):
        return (2 / self.N) * self.error
    
class BCE(Loss):
    """
    Binary Cross-Entropy Loss.
    Used for binary classification
    """
    def forward(self, y_pred, y_true):
        self.y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true
        self.N = y_pred.size
        
        # Calculate loss using the clipped predictions
        return (-1 / self.N) * np.sum(
            self.y_true * np.log(self.y_pred_clipped) + 
            (1 - self.y_true) * np.log(1 - self.y_pred_clipped)
        )
        
    def backward(self):
        return (1 / self.N) * ((self.y_pred_clipped - self.y_true) / 
                               (self.y_pred_clipped * (1 - self.y_pred_clipped)))

class CCE(Loss):
    """
    Categorical Cross-Entropy Loss.
    Used for multi-class classification (paired with Softmax).
    """
    def forward(self, y_pred, y_true):
        self.y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true
        
        self.N = y_pred.shape[0]
        
        return (-1 / self.N) * np.sum(self.y_true * np.log(self.y_pred_clipped))
        
    def backward(self):
        """
        If this is used immediately after Softmax (which passes grad_output unchanged),
        the combined gradient simplifies to just: (Prediction - True) / N
        """
        return (self.y_pred_clipped - self.y_true) / self.N