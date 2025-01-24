import numpy as np
from dynaconf import Dynaconf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.nn.initializers as init
from abc import ABC, abstractmethod
from Plants import Bathub_plant,Cournot_plant, FurnacePlant
from Controllers import Classic_controller, NN_controller

class System: 
    config = Dynaconf(settings_files=["config_file.json"])
    print(config.Plant)
    
    
    initial_state={}
    
    def initialize_system(self):
        
        print("Initialising Plant and Controller")
        
        plant_map = {
            "Bathtub": Bathub_plant,
            "Cournot": Cournot_plant,
            "Furnace": FurnacePlant
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
    
    def run_system(self):
        state=self.initial_state
        params=self.params
        error_list=[]
        param_list=[]
        self.target_list=[]

        
        gradfunc=jax.value_and_grad(self.run_one_epoch,argnums=0)
        for _ in range(self.config.Epochs):
            avg_error,gradients=gradfunc(params,state)
            # print("Gradients and error:", gradients,avg_error)

            params=self.Controller.update_params(params,gradients)
            state=self.reset_plant_and_error_history(state)
            error_list.append(avg_error)
            param_list.append(params)
        
        if (self.config.Visualization_flag):
            self.plot_target_list(self.target_list)
            self.plot_error_list(error_list)
            if (self.config.Controller == "Classic"):
                self.plot_param_list(param_list)
    
    def run_one_epoch(self,params,state):
        total_error=0
        control_signal=0
        output_list = []
        for _ in range(self.config.Simulation_steps_per_epoch):
            output = self.Plant.run_one_time_step(control_signal)
            error = self.target-output
            updated_params,control_signal,updated_state= self.Controller.run_one_time_step(params,state,error)
            params=updated_params
            state=updated_state

            total_error+=error**2  #Adds error squared to total error sum
            output_list.append(jnp.asarray(output).item())

        self.target_list.append(np.mean(output_list))
        return total_error/self.config.Simulation_steps_per_epoch #MSE
    
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
        
    def plot_target_list(self, target_list):
        """
        This function plots the error history during training.
        :param error_list: List of average error values.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(target_list, label="Target evolution")
        plt.xlabel("Epochs")
        plt.ylabel("Target")
        plt.title("Target history")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_param_list(self, param_list):
        """
        This function plots the param (kp,kd,ki) history during training.
        :param param_list: List of param values.
        """
        kp_vals = [params[0] for params in param_list]   
        kd_vals = [params[1] for params in param_list]   
        ki_vals = [params[2] for params in param_list]  

        plt.figure(figsize=(10, 6))
        plt.plot(kp_vals, label="kp", color='blue')  
        plt.plot(kd_vals, label="kd", color='green')  
        plt.plot(ki_vals, label="ki", color='red')  
        
        plt.xlabel("Epochs")
        plt.ylabel("Parameter Value")
        plt.title("Parameter History Over Epochs")
        plt.legend()  
        plt.grid(True)
        plt.show()
        


#START SYSTEM
system = System()
system.initialize_system()
system.run_system()
