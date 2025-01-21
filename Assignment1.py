import numpy as np
from dynaconf import Dynaconf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.nn.initializers as init
from abc import ABC, abstractmethod

#ABC abstract method, python debugger

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
    
class Plant(ABC):
    @abstractmethod
    def produce_initial_state(self):
        pass
    
    @abstractmethod
    def run_one_time_step(self,control_signal):
        pass
    
    @abstractmethod
    def reset(self):
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
            result_for_node = jax.nn.relu(result_for_node)
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
        
        
class Bathub_plant(Plant):

    def reset(self):
        self.H=self.initial_height     
        self.volume= self.initial_height*self.A
        
        
    def produce_initial_state(self):
        config = Dynaconf(settings_files=["config_file.json"])
        self.A=config.Bathtub_area
        self.C=config.Bathtub_cross_sectional_area
        self.initial_height=config.Bathtub_height
        self.H=self.initial_height
        self.g=config.Bathtub_gravity
        self.D_range=config.Disturbance_range
        self.V=np.sqrt(2*self.g*self.H)
        self.Q=self.V*self.C
        self.volume=self.H*self.A
            
    def get_volume(self, U, D):
        return U+D-self.Q
    
    def run_one_time_step(self,control_signal):
        D=np.random.uniform(self.D_range[0],self.D_range[1])
        velocity = jnp.sqrt(2*self.g*self.H)
        self.volume+=control_signal+D-self.Q
        self.H+=(control_signal+D-velocity/self.A)
        return self.H
                

class Cournot_plant(Plant):
    
    def produce_initial_state(self):
        config = Dynaconf(settings_files=["config_file.json"])
        self.max_price = config.Cournot_max_price
        self.marginal_cost = config.Cournot_marginal_cost
        self.D_range = config.Disturbance_range
        self.q1 = 0.8
        self.q2 = 0.5
    
    def get_price(self,q):
        return self.max_price-q
    
    def run_one_time_step(self,control_signal):
        D=np.random.uniform(self.D_range[0],self.D_range[1])
        q1 = jnp.clip(self.q1 + control_signal, 0, 1)
        q2 = jnp.clip(self.q2 + D, 0, 1)
        # q1 = self.q1 + control_signal  # Remove clipping temporarily
        # q2 = self.q2 + D  # Remove clipping temporarily
        # print("Here are the thingis",q1,q2)
        
        price = self.max_price-(q1+q2)
        # print("price", price)
        if price < 0: 
            print(f"Warning: Price went negative. Price={price}, q1={self.q1}, q2={self.q2}")
        self.q1=q1
        self.q2=q2
        return q1*(price-self.marginal_cost)
        
    def reset(self):
        self.produce_initial_state()
    
    

    
class System: 
    config = Dynaconf(settings_files=["config_file.json"])
    print(config.Plant)
    
    
    initial_state={}
    
    def initialize_system(self):
        ## HER KAN DET FORANDRES, bare kall hver greie med produce initial state og der leser man config filen!
        # NB! Kan man ha initial state her? Det er vel egtttt en del av kontrolleren!
        if(self.config.Plant=="Bathtub"):
            self.Plant=Bathub_plant()
            self.Plant.produce_initial_state()
            self.target=self.config.Bathtub_height
        elif self.config.Plant=="Cournot":
            self.Plant=Cournot_plant()
            self.Plant.produce_initial_state()
            self.target=self.config.Cournot_target
        if(self.config.Controller=="Classic"):
            self.Controller=Classic_controller()   
            self.Controller.produce_initial_state() 
            self.initial_state={"error_history":jnp.array([]),"kp": jnp.float32(0.5),"kd":jnp.float32(0.5),"ki":jnp.float32(0.5)}
            self.params = jnp.array([self.initial_state["kp"], self.initial_state["kd"], self.initial_state["ki"]], dtype=jnp.float32) #Denne kan kanskje flyttes oppover Må nok flyttes oppover også byttes med 
        if (self.config.Controller=="Neural_network"):
            self.Controller=NN_controller()
            self.params = self.Controller.produce_initial_state()    
            self.initial_state={"error_history":jnp.array([])}
            # for i 
            # self.params = jnp.array([state["kp"], state["kd"], state["ki"]], dtype=jnp.float32) #Denne kan kanskje flyttes oppover Må nok flyttes oppover også byttes med 
            
        print("HEIIIII")
        # init all components
    
    def run_system(self):
        state=self.initial_state
        params=self.params
        error_list=[]
        param_list=[]

        
        gradfunc=jax.value_and_grad(self.run_one_epoch,argnums=0)
        for _ in range(self.config.Epochs):
            avg_error,gradients=gradfunc(params,state)
            print("Gradients and error:", gradients,avg_error)
            # h= (self.target-avg_error)
            # print(h)
            # for i, (w_grad, b_grad) in enumerate(gradients):
            #     print(f"Layer {i}:")
                
            #     # Convert gradients to numpy arrays for better formatting
            #     w_grad_np = np.array(w_grad)
            #     b_grad_np = np.array(b_grad)

            #     # Use numpy's array2string for high-precision output
            #     print(f"Weight gradients:\n{np.array2string(w_grad_np, precision=20, floatmode='fixed')}")
            #     print(f"Bias gradients:\n{np.array2string(b_grad_np, precision=20, floatmode='fixed')}")

            #     # Check if gradients are effectively zero
            #     threshold = 1e-18
            #     w_zero_check = jnp.abs(w_grad) < threshold
            #     b_zero_check = jnp.abs(b_grad) < threshold

            #     print(f"Are weight gradients effectively zero? {w_zero_check}")
            #     print(f"Are bias gradients effectively zero? {b_zero_check}")

            #     # Check minimum and maximum gradient values
            #     print(f"Min weight gradient: {jnp.min(w_grad)}")
            #     print(f"Max weight gradient: {jnp.max(w_grad)}")
            #     print(f"Min bias gradient: {jnp.min(b_grad)}")
            #     print(f"Max bias gradient: {jnp.max(b_grad)}")


            params=self.Controller.update_params(params,gradients)
            print("UPDATED PARAMS: ",params)
            state=self.reset_plant_and_error_history(state)
            error_list.append(avg_error)
            param_list.append(params)
            
            
        self.plot_error_list(error_list)
        self.plot_param_list(param_list)
    
    def run_one_epoch(self,params,state):
        total_error=0
        control_signal=5
        
        for _ in range(self.config.Simulation_steps_per_epoch):
            error = self.target-self.Plant.run_one_time_step(control_signal)
            updated_params,control_signal,updated_state= self.Controller.run_one_time_step(params,state,error)
            # print(error, control_signal)
            params=updated_params
            state=updated_state

            # print(updated_state)
            total_error+=error**2
        return total_error/self.config.Simulation_steps_per_epoch
    
    def reset_plant_and_error_history(self,state):
        self.Plant.reset()
        state["error_history"]=jnp.array([])
        return state
    
    def plot_error_list(self, error_list):
        """
        This function plots the error history during training.
        :param error_list: List of average error values.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(error_list, label="Average Error")
        plt.xlabel("Epochs")
        plt.ylabel("Error (MSE)")
        plt.title("Error History Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_param_list(self, param_list):
        """
        This function plots the param (kp,kd,ki) history during training.
        :param param_list: List of param values.
        """
        kp_vals = [params[0] for params in param_list]  # Extracting the kp values
        kd_vals = [params[1] for params in param_list]  # Extracting the kd values
        ki_vals = [params[2] for params in param_list]  # Extracting the ki values

        plt.figure(figsize=(10, 6))
        plt.plot(kp_vals, label="kp", color='blue')  # Plotting kp values
        plt.plot(kd_vals, label="kd", color='green')  # Plotting kd values
        plt.plot(ki_vals, label="ki", color='red')  # Plotting ki values
        
        plt.xlabel("Epochs")
        plt.ylabel("Parameter Value")
        plt.title("Parameter History Over Epochs")
        plt.legend()  # Automatically handles the legend
        plt.grid(True)
        plt.show()
        


#START SYSTEM
system = System()
system.initialize_system()
system.run_system()
# NN=NN_controller()
# params = NN.produce_initial_state()
# initial_state={"error_history":jnp.array([])}
# a,b,c=NN.run_one_time_step(params,initial_state,5.0)
# print(b)
# print(NN.forward_pass(params, 0, np.array([1,2,3])))
