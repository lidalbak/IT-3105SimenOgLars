import numpy as np
from dynaconf import Dynaconf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.nn.initializers as init
from abc import ABC, abstractmethod

class Controller(ABC): 
    @abstractmethod
    def produce_initial_state(self):
        pass
    
    @abstractmethod
    def run_one_time_step(self, params,state,error):
        pass
    
    @abstractmethod
    def update_params(self,params,gradients):
        pass 
    
class Classic_controller(Controller): 
    
    def produce_initial_state(self):
        '''
        Produces initial state
        returns params
        '''
        config = Dynaconf(settings_files=["config_file.json"])
        self.Learning_rate=config.Learning_rate
        params = jnp.array([jnp.float32(config.initial_kp), jnp.float32(config.initial_kd), jnp.float32(config.initial_ki)], dtype=jnp.float32) 
        return params
    
    
    def run_one_time_step(self,params,state,error):
        '''
        Returns U (control signal)
        '''
        error_history=state.get("error_history")
        kp=params[0]
        kd=params[1]
        ki=params[2]
        
        if len(error_history) > 0:
            derivative_error = error - error_history[-1]
            integral_error = jnp.sum(error_history)
        else:
            derivative_error = 0
            integral_error = 0
        
        updated_error_history= jnp.concatenate([error_history, jnp.array([error])])
        
        updated_state = {
            **state,
            "error_history": updated_error_history,
        }
        return params,kp*error+kd*derivative_error+ki*integral_error,updated_state
    
    def update_params(self,params,gradients):
        '''
        Updates params
        '''
        learning_rate = self.Learning_rate
        updated_params = params - learning_rate * gradients
        return updated_params
        

class NN_controller(Controller):
    
    def produce_initial_state(self):
        '''
        Produces initial state
        returns Neural network
        '''
        config = Dynaconf(settings_files=["config_file.json"])
        self.Learning_rate=config.Learning_rate
        random_key = jax.random.PRNGKey(config.Random_seed) 
        activation_function = config.Neural_network_activation_function
        
        activation_map = {
            "relu": jax.nn.relu,
            "sigmoid": jax.nn.sigmoid,
            "tanh": jax.nn.tanh
        }
        
        if activation_function in activation_map:
            self.activation_function = activation_map[activation_function]
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
        
        first_layer_input_size = 3
        last_layer_output_size = 1
        
        layer_sizes=[first_layer_input_size] 
        for i in range(len(config.Neural_network_layers)):
            print("goes here")
            layer_sizes.append(config.Neural_network_layers[i])
        layer_sizes.append(last_layer_output_size)
        
        layers=[]
        
        for i in range(len(layer_sizes)-1):
            random_key, subkey_weights, subkey_biases = jax.random.split(random_key, 3)

            input_size=layer_sizes[i]
            output_size=layer_sizes[i+1]

            weights = jax.random.uniform(subkey_weights, shape=(output_size,input_size), minval=config.Neural_network_param_range_weights[0], maxval=config.Neural_network_param_range_weights[1]) 
            biases = jax.random.uniform(subkey_biases, shape=(output_size), minval=config.Neural_network_param_range_bias[0],maxval=config.Neural_network_param_range_bias[1])
           
            layers.append((weights, biases))

        return layers
        
    def forward_pass(self,params, layer_index, values):
        '''
        Values are the values passed on from the previous layer
        returns result,params
        '''
        result = jnp.array([])
        weights, biases = params[layer_index]
        for i in range(len(weights)):
            weights_for_node = weights[i]
            bias_for_node = biases[i]
            
            result_for_node = jnp.dot(values,weights_for_node) + bias_for_node
            result_for_node = self.activation_function(result_for_node)
            updated_result = jnp.concatenate([result, jnp.array([result_for_node])])
            
            result = updated_result
        return result,params
    
    def run_one_time_step(self,params,state,error):
        '''
        Runs one timestep
        returns params,prediction,state
        '''
        
        error_history = state.get("error_history")
        
        if len(error_history) > 0:
            derivative_error = error - error_history[-1]
            integral_error = jnp.sum(error_history)
        else:
            derivative_error = 0
            integral_error = 0
        
        updated_error_history= jnp.concatenate([error_history, jnp.array([error])])
        
        updated_state = {
            **state,
            "error_history": updated_error_history,
        }
        
        values=jnp.array([error,derivative_error,integral_error])
        for i in range(len(params)):
            updated_values, updated_params = self.forward_pass(params,i,values)
            params = updated_params
            values = updated_values
            
        
        return params,values[0],updated_state

    
    def update_params(self,params,gradients):
        updated_params = [
            (w - self.Learning_rate * grad_w, b - self.Learning_rate * grad_b) 
            for (w, b), (grad_w, grad_b) in zip(params, gradients)
        ]
        
        return updated_params
        