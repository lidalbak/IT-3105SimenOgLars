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
    def get_target(self):
        config = Dynaconf(settings_files=["config_file.json"])

        return config.Bathtub_height

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
    
    def get_target(self):
        config = Dynaconf(settings_files=["config_file.json"])

        return config.Cournot_target
        
    def reset(self):
        self.produce_initial_state()
    
    