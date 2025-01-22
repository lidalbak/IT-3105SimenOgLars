import numpy as np
from dynaconf import Dynaconf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.nn.initializers as init
from abc import ABC, abstractmethod
from Plants import Bathub_plant,Cournot_plant
from Controllers import Classic_controller, NN_controller

class System: 
    config = Dynaconf(settings_files=["config_file.json"])
    print(config.Plant)
    
    
    initial_state={}
    
    def initialize_system(self):
        plant_map = {
            "Bathtub": Bathub_plant,
            "Cournot": Cournot_plant,
        }

        controller_map = {
            "Classic": Classic_controller,
            "Neural_network": NN_controller,
        }

        #Plant
        plant_class = plant_map.get(self.config.Plant)
        self.Plant=plant_class()
        self.Plant.produce_initial_state()
        self.target = self.Plant.get_target()
        
        #Controller
        controller_class = controller_map.get(self.config.Controller)
        self.Controller = controller_class()
        self.params = self.Controller.produce_initial_state()
        self.initial_state={"error_history":jnp.array([])}
    
        
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
            self.initial_state={"error_history":jnp.array([])}
            self.params = self.Controller.produce_initial_state() 
        if (self.config.Controller=="Neural_network"):
            self.Controller=NN_controller()
            self.params = self.Controller.produce_initial_state()    
            self.initial_state = {"error_history":jnp.array([])}
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
