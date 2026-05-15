class Network:
    """
    The Conductor of our neural network. 
    It holds the layers, the loss function, and the optimizer, 
    and orchestrates the entire training process.
    """
    def __init__(self, layers, loss_fn, optimizer):
        # FIX 1: We must save the layers passed into the network!
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, input_data):
        """
        Passes the input through all layers sequentially.
        """
        result = input_data
        for layer in self.layers:
            result = layer.forward(result)
        return result 

    def backward(self):
        """
        Passes the gradient backward through all layers.
        """
        # We get the starting gradient from the loss function we just calculated
        grad_output = self.loss_fn.backward()
        
        # Pass it backward through the layers in reverse order
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    
    def train_step(self, x_batch, y_batch):
        """
        Orchestrates one full step of training:
        Forward -> Loss -> Backward -> Update
        """
        # 1. Forward Pass (Get predictions)
        predictions = self.forward(x_batch)
        
        # 2. Calculate Loss (How wrong were we?)
        error = self.loss_fn.forward(predictions, y_batch)
        
        # 3. Backward Pass (Calculate gradients)
        self.backward()
        
        # 4. Update (Tell the optimizer to adjust the weights)
        self.optimizer.update(self.layers)
        
        return error