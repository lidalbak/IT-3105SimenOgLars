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
    
        # self.state={"error_history":jnp.array([]),"kp": 1,"kd":1,"ki":1} # Skal være error history og hva kp, ki og ke er 

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
    
    def produce_initial_state(self):
        config = Dynaconf(settings_files=["config_file.json"])
        self.Learning_rate=config.Learning_rate
        params = jnp.array([jnp.float32(0.5), jnp.float32(0.5), jnp.float32(0.5)], dtype=jnp.float32) 
        return params
    
    def update_params(self,params,gradients):
        learning_rate = self.Learning_rate
        updated_params = params - learning_rate * gradients
        return updated_params
        

class NN_controller(Controller):
    ## NB neural net funker egt ikke tror æ, må flytte på den forward pass funksjonen inn i den one time stpe
    def produce_initial_state(self):
        config = Dynaconf(settings_files=["config_file.json"])
        self.Learning_rate=config.Learning_rate
        random_key = jax.random.PRNGKey(42) 
        first_input_size = 3
        last_output_size = 1
        layer_sizes=[first_input_size] #initialize with input layer size = 3
        for i in range(config.Neural_network_layers):
            layer_sizes.append(config.Neural_network_depth)
        layer_sizes.append(last_output_size)
        
        
        # self.network = jnp.array([[w00,w01,w02],[w10,w11,w12],[w20,w21,w22]])
        layers=[]
        
        for i in range(len(layer_sizes)-1):
            input_size=layer_sizes[i]
            output_size=layer_sizes[i+1]

            weights = jax.random.uniform(random_key, shape=(output_size,input_size), minval=config.Neural_network_param_range_weights[0], maxval=config.Neural_network_param_range_weights[1]) 
            biases = jax.random.uniform(random_key, shape=(output_size), minval=config.Neural_network_param_range_bias[0],maxval=config.Neural_network_param_range_bias[1])
           
            
            layers.append((weights, biases))

        
        # print(layers)
        return layers
        
    def forward_pass(self,params, layer_index, values):
        '''
        Values are the values passed on from the previous layer
        '''
        result = jnp.array([])
        weights, biases = params[layer_index]
        for i in range(len(weights)):
            weights_for_node = weights[i]
            bias_for_node = biases[i]
            # print("VAL=",values)
            # print("Weights=",weights_for_node)
            result_for_node = jnp.dot(values,weights_for_node) + bias_for_node
            # result_for_node = jax.nn.relu(result_for_node)
            result = jnp.concatenate([result, jnp.array([result_for_node])])
            # result.append(result_for_node)
            # print("WTTTTT",weights)
            # weights = layer.get["weights"]
            # biases = layer.get["biases"]
            # for 
        
            # NB make all arrays to jnp 
        
        # print(values[0])
        # print("Forward_ pass result:",result)
        return result,params
    
    def run_one_time_step(self,params,state,error):
        
        error_history=state.get("error_history")
        
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
            (w - self.Learning_rate * grad_w, b - self.Learning_rate * grad_b)  # Update weights and biases for each layer
            for (w, b), (grad_w, grad_b) in zip(params, gradients)
        ]
        
        return updated_params
        