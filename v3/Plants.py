import numpy as np
from dynaconf import Dynaconf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.nn.initializers as init
from abc import ABC, abstractmethod

    
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
    
    @abstractmethod
    def get_target(self):
        pass



class Bathub_plant(Plant):

    def reset(self):
        '''
        Reset plant
        '''
        self.H=self.initial_height     
        self.volume= self.initial_height*self.A
        
        
    def produce_initial_state(self):
        '''
        Produces initial state
        '''
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
    
    def run_one_time_step(self,control_signal):
        '''
        Runs one timestep
        '''
        D=np.random.uniform(self.D_range[0],self.D_range[1])
        
        velocity = jnp.sqrt(2*self.g*self.H)
        self.volume+=control_signal+D-self.Q
        self.H+=(control_signal+D-velocity/self.A)
        
        return self.H
    
    def get_target(self):
        '''
        Returns target attribute for plant
        '''
        config = Dynaconf(settings_files=["config_file.json"])

        return config.Bathtub_height

class Cournot_plant(Plant):
    
    def produce_initial_state(self):
        '''
        Prouduces initial state
        '''
        config = Dynaconf(settings_files=["config_file.json"])
        self.max_price = config.Cournot_max_price
        self.marginal_cost = config.Cournot_marginal_cost
        self.D_range = config.Disturbance_range
        self.q1 = config.initial_q1
        self.q2 = config.initial_q2
    
    
    def run_one_time_step(self,control_signal):
        '''
        Runs one time step
        returns q1 profit
        '''
        D=np.random.uniform(self.D_range[0],self.D_range[1])
        q1 = jnp.clip(self.q1 + control_signal, 0, 1)
        q2 = jnp.clip(self.q2 + D, 0, 1)
    
        price = self.max_price-(q1+q2)
       
        if price < 0: 
            print(f"Warning: Price went negative. Price={price}, q1={self.q1}, q2={self.q2}")
            
        self.q1=q1
        self.q2=q2
        return q1*(price-self.marginal_cost)
    
    def get_target(self):
        '''
        Returns target attribute value
        '''
        config = Dynaconf(settings_files=["config_file.json"])

        return config.Cournot_target
        
    def reset(self):
        '''
        Resets state
        '''
        self.produce_initial_state()
        
class FurnacePlant:
    def reset(self):
        """
        Resets the furnace to its initial temperature.
        """
        self.T = self.initial_temperature

    def produce_initial_state(self):
        """
        Produces initial state
        """
        config = Dynaconf(settings_files=["config_file.json"])
        self.C = config.Furnace_heat_capacity  
        self.h = config.Furnace_heat_loss_coefficient  
        self.T_env = config.Ambient_temperature  
        self.initial_temperature = config.Furnace_initial_temperature  
        self.T = self.initial_temperature  
        self.D_range = config.Disturbance_range  

    def get_temperature(self, U, D):
        """
        Calculates the new temperature based on input heat, disturbances, and heat losses.
        """
        dT_dt = (U + D - self.h * (self.T - self.T_env)) / self.C
        return dT_dt

    def run_one_time_step(self, control_signal):
        """
        Runs a single time step
        returns Temprature
        """
        D = np.random.uniform(self.D_range[0], self.D_range[1])
        
        dT_dt = self.get_temperature(control_signal, D)
        self.T += dT_dt 
        
        return self.T

    def get_target(self):
        """
        Returns the target temperature for the furnace.
        """
        config = Dynaconf(settings_files=["config_file.json"])
        return config.Furnace_target_temperature

    
    