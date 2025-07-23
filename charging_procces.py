import simpy    

class EVChargingProcess:
    def __init__(self, env, ev, charger_resource, load_curve, time_step, simulation_setup, charger_instance, done_event):
        self.env = env
        self.ev = ev
        self.charger_resource = charger_resource  # Now always None
        self.load_curve = load_curve
        self.time_step = time_step
        self.simulation_setup = simulation_setup
        self.charger_instance = charger_instance
        self.done_event = done_event
        self.action = env.process(self.run())

    def run(self):
        # No need to calculate queue time here - it's done in the charger process
        # Add to charging pool and wait for done_event
        self.simulation_setup.charging_pool.append((self, self.charger_instance, self.done_event))
        yield self.done_event  # Wait until coordinator triggers completion
        # At this point, EV is fully charged and can release the charger