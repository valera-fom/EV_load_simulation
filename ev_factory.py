from EV import EV, EV_MODELS
import pandas as pd
import numpy as np
import simpy
from charger import Charger, CHARGER_MODELS

def create_ev(model_name, soc=0.2):
    if model_name not in EV_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    EV_specs = EV_MODELS[model_name]

    return EV(name=model_name,
              battery_capacity=EV_specs["capacity"],
              max_charging_power=EV_specs["max_power"],
              soc=soc)

def create_charger(env, model_name, num_ports=1):
    if model_name not in CHARGER_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    charger_specs = CHARGER_MODELS[model_name]

    return Charger(env, 
                name=model_name,
                power_kW=charger_specs["power_kW"],
                num_ports=num_ports)




