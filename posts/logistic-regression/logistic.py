import torch


class LinearModel:

    def __init__(self):
        self.w = None 
        self.previousWeight = None
        
    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))


        #compute the vector of scores s
        return X@(self.w)


    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s = self.score(X)
        y_hat = (s > 0.0) * 1.0
        return y_hat


class LogisticRegression(LinearModel):

    
    def loss(self, X, y):
        """
        Compute the empirical risk L(w) using the logistic loss function

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix
            
            y, torch.Tensor: the vector of target labels (0 or 1). y.size() = (n,)

        RETURNS: 
            loss torch.Tensor: float: loss L(w)
        """
        
        s = self.score(X)
        
        sig = 1.0/(1.0+torch.exp(-s))


        sig[sig == 1.0] = 0.9999999
        sig[sig == 0.0] = 0.0000001
        
        return torch.mean(-y*torch.log(sig) - (1 - y)*torch.log(1-sig))
    


    def grad(self, X, y):
        """
        Computes the gradient of the empirical risk L(w)

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            
            y, torch.Tensor: the vector of target labels (0 or 1). y.size() = (n,)

        RETURNS: 
            torch.Tensor: float: gradient of the empirical risk
        """
        
        s = self.score(X)
        
        sig = 1/(1+torch.exp(-s))

        sig[sig == 1.0] = 0.9999999
        sig[sig == 0.0] = 0.0000001
        
        return torch.mean((sig - y)[:, None]  * X, dim = 0)




class GradientDescentOptimizer(LogisticRegression):
    
    def __init__(self, model):
        self.model = model
        
    def step(self, X, y, alpha, beta):
        """
         Implements spicy gradient descent with momentum. Updates the weight of the model.

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            
            y, torch.Tensor: the vector of target labels (0 or 1). y.size() = (n,)
            
            alpha, float: the learning rate
            beta, float: the momentum parameter

        RETURNS: 
            torch.Tensor: float: the loss L(w)
        """
        weight = self.model.w 

        gradient = self.model.grad(X, y)
        

        if(self.model.previousWeight != None):
            self.model.w = weight - alpha * gradient + beta * (weight - self.model.previousWeight)
            self.model.previousWeight = self.model.w
        else:   
            self.model.previousWeight = torch.rand(X.size()[1])
                

        return self.model.loss(X, y)