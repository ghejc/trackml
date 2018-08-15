import numpy as np

#Imports of the Keras library parts we will need
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Dropout, Concatenate
from keras import optimizers

#Import of TensorFlow backend of Keras
from keras import backend as K

class GaussianMixtureDensityNetwork:
    """"
    This class learns a conditional probability distribution of the form p(y|X) from given data
    using a Keras neural network.
    
    Based on an implementation by Axel Brando see
        https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation
    
    The changes include a porting from Keras 1.1 to Keras 2 and restructuring the code to a class
    with a similar API as a scikits-learn regressor.
    """
    
    def __init__(self, input_dim, output_dim, num_distributions=100, num_hidden_layers=3, num_hidden_units=512, dropout=0.25, activation='relu', optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)):
        """ 
        Creates a Keras model for fitting the parameters mu,sigma,alpha of a Gaussian mixture distribution
        to the input data X and y
        The argument input_dim is the axis=0 dimension of X.
        The argument output_dim is the dimension of y (or mu).
        The number of outputs is output_dim + 2.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_distributions = num_distributions
        self.optimizer = optimizer
        inputs = Input(shape = (input_dim,))
        x = inputs

        for _ in range(num_hidden_layers):
            # create hidden layer
            x = Dense(units = num_hidden_units, activation = activation)(x)
            x = Dropout(dropout)(x)

        # final hidden layer
        x = Dense(units = num_hidden_units, activation = activation)(x)
        # mixture density parameters
        mu = Dense(units = output_dim*num_distributions)(x)
        sigma = Dense(units = num_distributions, activation = GaussianMixtureDensityNetwork.elu_modif)(x) #K.exp, W_regularizer=l2(1e-3)
        alpha = Dense(units = num_distributions, activation = 'softmax')(x)
        outputs = Concatenate(axis=1)([mu, sigma, alpha]) 
        self.model = Model(inputs = inputs, outputs = outputs)
        
        self.model.compile(optimizer = self.optimizer, loss = self.scoring_rule_adv)
        
    @staticmethod
    def log_sum_exp(x, axis=None):
        """ Log-sum-exp trick implementation"""
        x_max = K.max(x, axis=axis, keepdims=True)
        return K.log(K.sum(K.exp(x - x_max), 
                       axis=axis, keepdims=True)) + x_max

    @staticmethod
    def elu_modif(x, a=1.):
        """ Modified ELU activation function """
        e = 1e-15
        return K.elu(x,alpha=a)+1.+e
    
    def log_Gaussian_likelihood(self, y, parameters):
        """ Log Gaussian Likelihood distribution """
        components = K.reshape(parameters,[-1, self.output_dim + 2, self.num_distributions])
        mu = components[:, :self.output_dim, :]
        sigma = components[:, self.output_dim, :]
        alpha = components[:, self.output_dim + 1, :]
        alpha = K.softmax(K.clip(alpha,1e-8,1.))
    
        exponent = K.log(alpha) - .5 * float(self.output_dim) * K.log(2 * np.pi) \
        - float(self.output_dim) * K.log(K.abs(sigma)) - \
        K.sum((K.expand_dims(y,2) - mu)**2, axis=1)/(2*(sigma)**2)
    
        log_gauss = GaussianMixtureDensityNetwork.log_sum_exp(exponent, axis=1)
        return log_gauss
    
    def Gaussian_argmax(self, y0, parameters):
        """
        x = Sum(a[i]*mu[i]/sigma[i]^2*exp(-(x-mu[i])^2/(2*sigma[i]^2)),i = 1..num_distributions)/
            Sum(a[i]/sigma[i]^2*exp(-(x-mu[i])^2/(2*sigma[i]^2)),i = 1..num_distributions)      
        """
        components = K.reshape(parameters,[-1, self.output_dim + 2, self.num_distributions])
        mu = components[:, :self.output_dim, :]
        sigma = components[:, self.output_dim, :]
        alpha = components[:, self.output_dim + 1, :]
        alpha = K.softmax(K.clip(alpha,1e-8,1.))
    
        exponent = K.log(alpha) - .5 * float(self.output_dim) * K.log(2 * np.pi) \
        - float(self.output_dim + 2) * K.log(K.abs(sigma)) + \
        K.sum((K.expand_dims(y0,2) - mu)**2, axis=1)/(2*(sigma)**2)

        log_Z = GaussianMixtureDensityNetwork.log_sum_exp(exponent, axis=1)
        func = lambda i : K.sum(mu[:,i,:] * K.exp(exponent - log_Z), axis=1, keepdims=True)
        return K.map_fn(func, K.arange(0, self.output_dim), dtype='float32')

    def mean_log_Gaussian_likelihood(self, y_true, parameters):
        """ Mean Log Gaussian Likelihood distribution
        """
        return -K.mean(self.log_Gaussian_likelihood(y_true, parameters))

    def scoring_rule_adv(self, y_true, y_pred):
        """ Fast Gradient Sign Method (FSGM) to implement Adversarial Training
        """
        # Compute loss 
        error = self.mean_log_Gaussian_likelihood(y_true, y_pred)
    
        # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
        # Define gradient of loss wrt input
        grad_error = K.gradients(error,self.model.input) #Minus is on error function
        # Take sign of gradient, Multiply by constant epsilon,
        # Add perturbation to original example to obtain adversarial example
        # Sign add a new dimension we need to obviate
    
        epsilon = 0.08
    
        adversarial_X = K.stop_gradient(self.model.input + epsilon * K.sign(grad_error)[0])
    
        # Note: If you want to test the variation of adversarial training 
        #  proposed by XX, eliminate the following comment character 
        #  and comment the previous one.
    
        ##adversarial_X = self.model.input + epsilon * K.sign(grad_error)[0]
    
        adv_output = self.model(adversarial_X)
    
        adv_error = self.mean_log_Gaussian_likelihood(y_true, adv_output)
        return 0.3 * error + 0.7 * adv_error
    
    def fit(self, X, y, batch_size=10000, epoch=200, validation_split=0.1):
        """ Fits model to data
        """
        self.model.fit(X, y, batch_size = batch_size,
                       nb_epoch = epoch, validation_split = validation_split)
        
    def partial_fit(self, X, y):
        """ Fits the model to batches of data
        """
        self.model.train_on_batch(X, y)
        
    def predict_log_proba(self, X, y):
        """ Predicts the log probability of y given X
        """
        X = X.reshape((-1, self.input_dim))
        parameters = self.model.predict(X)
        y = y.reshape((-1,self.output_dim))
        # check for nan
        if np.any(np.isnan(parameters)):
            return np.full((y.shape[0]), -np.inf)
        parameters = K.variable(parameters)
        x = K.variable(y)
        func = lambda i : self.log_Gaussian_likelihood(x[i:i+1,:], parameters)
        return K.eval(K.map_fn(func, K.arange(0, y.shape[0]),dtype='float32')).reshape((y.shape[0]))

    def predict(self, X, y0 = None):
        """
        Returns the argument with the highest log probability density
        for the given single input X closest to y0. If y0 is None, the
        start value is taken from the last elements of X.
        """
        X = X.reshape((-1, self.input_dim))
        parameters = self.model.predict(X)
        if y0 is None:
            y0 = X[:,-self.output_dim:]
        y0 = y0.reshape((-1,self.output_dim))
        # check for nan
        if np.any(np.isnan(parameters)):
            return y0.reshape((-1))
        parameters = K.variable(parameters)
        y = K.variable(y0)
        return K.eval(self.Gaussian_argmax(y, parameters)).reshape((-1))

    def save(self, file_name):
        "Save model and weights to a file. File name is without extension."
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(file_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(file_name + '.h5df')

    def load(self, file_name):
        "Load model and weights from a file. File name is without extension."
        # serialize model to JSON
        json_file = open(file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(file_name + '.h5df')
        loaded_model.compile(optimizer = self.optimizer, loss = self.scoring_rule_adv)
        self.model = loaded_model
        