class Charger:
    def __init__(self, env, name, power_kW, num_ports):
        self.env = env
        self.name = name
        self.power_kW = power_kW
        self.num_ports = num_ports

CHARGER_MODELS = {
    
    "ac": {
        "type": "AC",
        "power_kW": 11
    },
    "dc": {
        "type": "DC",
        "power_kW": 50,
    },
    "fast_dc": {
        "type": "DC",
        "power_kW": 200,
    }
}


