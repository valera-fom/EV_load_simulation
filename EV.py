class EV:
    def __init__(self, name, battery_capacity, max_charging_power, soc=0.2):
        self.name = name
        self.battery_capacity = battery_capacity
        self.max_charging_power = max_charging_power
        self.soc = soc

    @property
    def energy_needed(self):
        return self.battery_capacity * (1 - self.soc)

EV_MODELS = {
    # Passenger BEVs
    "bev_small": 
        {
            "name": "Renault Zoe",
            "capacity": 52,
            "DC": 52,
            "AC": 22
        }
    ,
    "bev_medium": 
        {
            "name": "VW ID.3",
            "capacity": 60,
            "DC": 120,
            "AC": 11
        }
    ,
    "bev_big": 
        {
            "name": "Tesla Model 3",
            "capacity": 78.1,
            "DC": 250,
            "AC": 11
        }
    ,

    # Passenger PHEVs
    "phev_small": 
        {
            "name": "BMW 330e",
            "capacity": 12,
            "DC": None,
            "AC": 3.7
        }
    ,
    "phev_medium": 
        {
            "name": "Peugeot 308 PHEV",
            "capacity": 12.4,
            "DC": None,
            "AC": 7.4
    },
    "phev_big": {
            "name": "Volvo XC60 Recharge",
            "capacity": 18.8,
            "DC": None,
            "AC": 6.6    
    },

    # Light Commercial BEVs
    "commercial_bev_small": {
            "name": "Renault Kangoo E-Tech",
            "capacity": 45,
            "DC": 80,
            "AC": 11
        }
    ,
    "commercial_bev_medium": {
            "name": "Ford E-Transit",
            "capacity": 67,
            "DC": 115,
            "AC": 11
        }
    ,
    "commercial_bev_big": {
            "name": "Peugeot e-Expert",
            "capacity": 75, 
            "DC": 100,
            "AC": 11
        }
    ,

    # Heavy Vehicles BEVs
    "heavy_bev_small": {
            "name": "Volvo FH Electric",
            "capacity": 540,
            "DC": 250,
            "AC": None
        }
    ,
    "heavy_bev_medium": {
            "name": "Mercedes eActros LongHaul",
            "capacity": 621,
            "DC": 400,
            "AC": None
        }
    ,
    "heavy_bev_big": {
            "name": "Scania 40R BEV",
            "capacity": 624,
            "DC": 375,
            "AC": None
        }
    }



