import simpy
import numpy as np
from EV import EV, EV_MODELS
from charger import Charger, CHARGER_MODELS
from charging_procces import EVChargingProcess
from ev_factory import create_charger
import random
import os

def clear_console():
    """Clear console output to prevent memory buildup from old simulation logs."""
    os.system('cls' if os.name == 'nt' else 'clear')

class SimulationSetup:
    def __init__(self, ev_counts, charger_counts, sim_duration=24*60, time_step=1, 
                 arrival_time_mean=12*60, arrival_time_span=4*60, grid_power_limit=0, verbose=False):
        """
        ev_counts: dict of {ev_type: number_of_evs}
        charger_counts: dict of {charger_type: number_of_chargers}
        sim_duration: total simulation time in minutes (default 24h)
        time_step: time step in minutes (default 1 min)
        arrival_time_mean: mean arrival time in minutes (default 12h = noon)
        arrival_time_span: 2*sigma for normal distribution in minutes (default 4h)
        grid_power_limit: maximum grid power in kW (0 = no limit)
        verbose: enable detailed debug output (default False)
        """
        self.ev_counts = ev_counts
        self.charger_counts = charger_counts
        self.sim_duration = sim_duration
        self.time_step = time_step
        self.arrival_time_mean = arrival_time_mean
        self.arrival_time_span = arrival_time_span
        self.verbose = verbose
        self.grid_power_profile = None
        if isinstance(grid_power_limit, (list, np.ndarray)):
            self.grid_power_profile = np.array(grid_power_limit)
            self.grid_power_limit = None
        else:
            self.grid_power_limit = grid_power_limit
            self.grid_power_profile = None
        self.env = simpy.Environment()
        self.chargers = {}  # charger_type: list of Charger objects
        self.charger_resources = {}  # charger_type: SimPy Resource
        self.evs = []  # List of EV objects
        self.load_curve = np.zeros(sim_duration // time_step)
        self.current_grid_power = 0  # Track current grid power usage
        self.charging_pool = []  # List of (EVChargingProcess, energy_needed, done_event)
        self.rejected_evs = []  # Track rejected EVs and reasons
        self.global_ev_queue = simpy.FilterStore(self.env)
        self._init_chargers()
        self._init_evs()
        self.env.process(self.charging_coordinator())
        self._start_charger_processes()
        
        # Debug: Check total charger count (only if verbose)
        if self.verbose:
            total_chargers = sum(len(charger_list) for charger_list in self.chargers.values())
            print(f"Total chargers initialized: {total_chargers}")
            if total_chargers > 8:
                print(f"WARNING: High charger count ({total_chargers}) - monitoring for resource issues")

    def _init_chargers(self):
        for charger_type, count in self.charger_counts.items():
            if count == 0:
                continue  # Skip creating resources with zero capacity
            self.chargers[charger_type] = []
            self.charger_resources[charger_type] = simpy.Resource(self.env, capacity=count)
            for i in range(count):
                charger_name = f"{charger_type}_{i+1}"
                charger = create_charger(self.env, charger_type, num_ports=1)
                charger.name = charger_name  # Ensure unique name
                # Add a resource constraint to each charger instance
                charger.resource = simpy.Resource(self.env, capacity=1)
                self.chargers[charger_type].append(charger)

    def _init_evs(self):
        for ev_type, count in self.ev_counts.items():
            ev_specs = EV_MODELS[ev_type]
            for i in range(count):
                ev_name = f"{ev_specs['name']}_{i+1}"
                # Prioritize DC over AC for charging power
                max_power = ev_specs.get('DC') or ev_specs.get('AC')
                ev = EV(name=ev_name, battery_capacity=ev_specs["capacity"], max_charging_power=max_power, soc=0.2)
                ev.ev_type = ev_type  # Store the EV type for later use
                self.evs.append(ev)

    def _generate_arrival_times(self):
        """
        Generate arrival times using normal distribution centered around the mean
        """
        total_evs = len(self.evs)
        sigma = self.arrival_time_span / 2  # Convert 2*sigma to sigma
        
        # Generate normal distribution arrival times centered around the mean
        arrival_times = np.random.normal(self.arrival_time_mean, sigma, total_evs)
        
        # Debug: Show distribution statistics (only if verbose)
        if self.verbose:
            print(f"    Arrival time distribution:")
            print(f"      Mean: {np.mean(arrival_times)/60:.1f} hours (target: {self.arrival_time_mean/60:.1f}h)")
            print(f"      Std: {np.std(arrival_times)/60:.1f} hours (target: {sigma/60:.1f}h)")
            print(f"      Range: {np.min(arrival_times)/60:.1f}h - {np.max(arrival_times)/60:.1f}h")
        
        # Ensure arrival times are within simulation bounds
        arrival_times = np.clip(arrival_times, 0, self.sim_duration - 60)  # Leave 1 hour buffer
        
        # Sort arrival times to ensure chronological order
        arrival_times.sort()
        
        return arrival_times

    def _start_charger_processes(self):
        total_processes = 0
        for charger_type, charger_list in self.chargers.items():
            for charger_instance in charger_list:
                self.env.process(self._charger_process(charger_instance, charger_type))
                total_processes += 1
        print(f"Started {total_processes} charger processes")
        
        # Debug: Check if we're hitting any limits
        if total_processes > 8:
            print(f"WARNING: High number of charger processes ({total_processes}) - monitoring for issues")

    def _charger_process(self, charger_instance, charger_type):
        while True:
            # Wait for a compatible EV in the global queue
            ev = yield self.global_ev_queue.get(lambda ev: self._is_compatible(ev, charger_type))
            
            # Debug: Print charger assignment
            print(f"Charger {charger_instance.name} ({charger_type}) assigned to {ev.name} (type: {ev.ev_type})")
            
            # Request the charger resource to ensure exclusive access
            with charger_instance.resource.request() as request:
                yield request
                
                # Calculate actual queue time
                ev.queue_time = self.env.now - ev.queued_at
                
                # Set the EV's max charging power for this charger, never None
                ev_type = ev.ev_type
                ev_specs = EV_MODELS[ev_type]
                if CHARGER_MODELS[charger_type]['type'] == 'AC':
                    ev.max_charging_power = ev_specs['AC'] if ev_specs['AC'] is not None else 0
                else:
                    ev.max_charging_power = ev_specs['DC'] if ev_specs['DC'] is not None else 0
                # Always use the minimum of EV and charger power
                ev.max_charging_power = min(ev.max_charging_power, charger_instance.power_kW)
                
                print(f"  {ev.name} charging at {ev.max_charging_power} kW on {charger_instance.name}")
                
                # Start charging process and wait for it to finish
                done_event = self.env.event()
                charging_process = EVChargingProcess(
                    self.env,
                    ev,
                    None,  # No resource needed, as the charger process is the resource
                    self.load_curve,
                    self.time_step,
                    self,
                    charger_instance,
                    done_event
                )
                # The charging process automatically adds itself to the pool in its run() method
                yield done_event  # Wait until charging is done before picking the next EV

    def _is_compatible(self, ev, charger_type):
        ev_type = ev.ev_type
        ev_specs = EV_MODELS[ev_type]
        charger_model = CHARGER_MODELS[charger_type]
        if charger_model['type'] == 'AC' and ev_specs.get('AC') is not None:
            return True
        if charger_model['type'] == 'DC' and ev_specs.get('DC') is not None:
            return True
        return False

    def schedule_evs(self):
        print(f"\nScheduling {len(self.evs)} EVs with random arrival times...")
        print(f"Arrival time mean: {self.arrival_time_mean/60:.1f} hours")
        print(f"Arrival time span (2σ): {self.arrival_time_span/60:.1f} hours")
        if self.grid_power_limit is not None:
            print(f"Grid power limit: {self.grid_power_limit} kW")
        else:
            print(f"Grid power profile: {self.grid_power_profile}")
        arrival_times = self._generate_arrival_times()
        scheduled_count = 0
        for i, ev in enumerate(self.evs):
            # Check for at least one compatible charger type
            compatible = False
            for charger_type in self.chargers:
                if self._is_compatible(ev, charger_type):
                    compatible = True
                    break
            if not compatible:
                reason = "No compatible charger available"
                self.rejected_evs.append((ev, reason))
                continue
            arrival_time = arrival_times[i]
            self.env.process(self._ev_arrival(ev, arrival_time))
            scheduled_count += 1
        print(f"Scheduled {scheduled_count} out of {len(self.evs)} EVs")
        self._display_rejection_notifications()

    def _ev_arrival(self, ev, arrival_time):
        yield self.env.timeout(arrival_time)
        # Store the time when EV is put into queue
        ev.queued_at = self.env.now
        # Put the EV in the global queue
        yield self.global_ev_queue.put(ev)
        # Debug: Track queue size
        queue_size = len(self.global_ev_queue.items)
        print(f"EV {ev.name} queued. Queue size: {queue_size}")

    def run_simulation(self):
        """
        Run the complete simulation.
        """
        # Clear console to prevent memory buildup from old logs
        if not self.verbose:
            clear_console()
        
        # Schedule EVs with arrival times
        self.schedule_evs()
        
        # Run the simulation
        print(f"\nStarting simulation for {self.sim_duration/60:.1f} hours...")
        self.env.run(until=self.sim_duration)
        print(f"Simulation completed at {self.env.now/60:.1f} hours\n")
        
        # Display rejection notifications
        self._display_rejection_notifications()
        
        # Display charger usage statistics
        self._display_charger_statistics()

    def charging_coordinator(self):
        """
        Central process that allocates grid power equally (with max cap and redistribution) to all charging EVs every minute.
        """
        while True:
            # Remove finished EVs from pool
            self.charging_pool = [item for item in self.charging_pool if not item[2].triggered]
            
            if self.charging_pool:
                sim_time = int(self.env.now)
                grid_profile = self.grid_power_profile
                grid_limit = self.grid_power_limit
                # Fix for NumPy array boolean ambiguity
                if grid_profile is not None and (not hasattr(grid_profile, '__len__') or len(grid_profile) > 0):
                    current_limit = grid_profile[sim_time] if sim_time < len(grid_profile) else grid_profile[-1]
                else:
                    current_limit = grid_limit if grid_limit is not None else float('inf')
                
                # Debug: Track grid limit behavior (only if verbose)
                if self.verbose:
                    print(f"  Grid limit: {current_limit:.2f} kW")
                
                # Debug: Track charger assignments and check for duplicates (only if verbose)
                if self.verbose:
                    charger_assignments = {}
                    for proc, charger_instance, done_event in self.charging_pool:
                        charger_id = charger_instance.name
                        if charger_id not in charger_assignments:
                            charger_assignments[charger_id] = []
                        charger_assignments[charger_id].append(proc.ev.name)
                    
                    # Debug: Print time step summary
                    total_evs_charging = len(self.charging_pool)
                    total_chargers = sum(len(charger_list) for charger_list in self.chargers.values())
                    print(f"\n[Time {sim_time}] EVs charging: {total_evs_charging}, Total chargers: {total_chargers}")
                    
                    # Debug: Check for multiple EVs on same charger
                    for charger_id, evs in charger_assignments.items():
                        if len(evs) > 1:
                            print(f"ERROR: Multiple EVs on charger {charger_id}: {evs}")
                        else:
                            print(f"  {charger_id}: {evs[0] if evs else 'None'}")
                
                pool = self.charging_pool.copy()
                N = len(pool)
                
                # Calculate the actual maximum possible charging capacity
                total_charger_capacity = sum(
                    len(charger_list) * CHARGER_MODELS[charger_type]['power_kW']
                    for charger_type, charger_list in self.chargers.items()
                )
                
                # Calculate total EV demand (sum of all EVs' max charging power)
                total_ev_demand = sum(
                    min(proc.ev.max_charging_power if proc.ev.max_charging_power is not None else 0, charger_instance.power_kW)
                    for proc, charger_instance, done_event in pool
                )
                
                # Use the minimum of: grid limit, total charger capacity, total EV demand
                effective_limit = min(current_limit, total_charger_capacity, total_ev_demand)
                
                # Debug output (only if verbose)
                if self.verbose:
                    print(f"  Grid limit: {current_limit:.2f} kW, Charger capacity: {total_charger_capacity:.2f} kW, EV demand: {total_ev_demand:.2f} kW")
                    print(f"  Effective limit: {effective_limit:.2f} kW")
                
                remaining_power = effective_limit
                allocations = [0.0] * N
                capped = [False] * N
                
                while remaining_power > 1e-6 and not all(capped):
                    share = remaining_power / sum(1 for i in range(N) if not capped[i])
                    for i, (proc, charger_instance, done_event) in enumerate(pool):
                        if capped[i]:
                            continue
                        # Always use the minimum of EV and charger power
                        max_power = min(proc.ev.max_charging_power if proc.ev.max_charging_power is not None else 0, charger_instance.power_kW)
                        alloc = min(share, max_power)
                        allocations[i] += alloc
                        if alloc >= max_power:
                            capped[i] = True
                        remaining_power -= alloc
                        if remaining_power < 1e-6:
                            break
                
                # Debug: Print allocation details and check total (only if verbose)
                if self.verbose:
                    total_allocated = sum(allocations)
                    print(f"  Total allocated: {total_allocated:.2f} kW")
                    for i, (proc, charger_instance, done_event) in enumerate(pool):
                        print(f"    {proc.ev.name} on {charger_instance.name}: {allocations[i]:.2f} kW (max: {proc.ev.max_charging_power}, charger: {charger_instance.power_kW})")
                    
                    # Check if total allocated exceeds total charger capacity
                    if total_allocated > total_charger_capacity:
                        print(f"ERROR: Total allocated ({total_allocated:.2f} kW) exceeds total charger capacity ({total_charger_capacity:.2f} kW)")
                    
                    # Check if we're hitting grid limit exactly
                    if abs(total_allocated - current_limit) < 0.1:
                        print(f"WARNING: Allocation exactly matches grid limit ({current_limit:.2f} kW) - possible snapping behavior")
                
                for i, (proc, charger_instance, done_event) in enumerate(pool):
                    if done_event.triggered:
                        continue
                    delivered = allocations[i] / 60.0
                    proc.ev.soc += delivered / proc.ev.battery_capacity
                    energy_needed = proc.ev.battery_capacity * (1 - proc.ev.soc)
                    
                    if proc.ev.soc >= 1.0 or energy_needed <= 0:
                        proc.ev.soc = 1.0
                        done_event.succeed()
                    
                    t = int(self.env.now // self.time_step)
                    if t < len(self.load_curve):
                        self.load_curve[t] += allocations[i]
            
            yield self.env.timeout(1)

    def _display_rejection_notifications(self):
        """
        Display information about rejected EVs and reasons.
        """
        if self.rejected_evs:
            print(f"\nRejected EVs ({len(self.rejected_evs)}):")
            for ev, reason in self.rejected_evs:
                print(f"  {ev.name}: {reason}")
        else:
            print("\n✅ All EVs scheduled successfully!")
    
    def _display_charger_statistics(self):
        """
        Display charger usage statistics.
        """
        print(f"\nCharger Usage Statistics:")
        for charger_type, charger_list in self.chargers.items():
            total_chargers = len(charger_list)
            print(f"  {charger_type}: {total_chargers} chargers")
        
        # Compute average queue time using only EVs that actually waited
        queue_times = [ev.queue_time for ev in self.evs if hasattr(ev, 'queue_time') and ev.queue_time > 0]
        avg_queue_time = np.mean(queue_times) if queue_times else 0
        print(f"Average queue time (for {len(queue_times)} EVs that waited): {avg_queue_time:.2f} minutes")
        self.avg_queue_time = avg_queue_time
        self.num_queued_evs = len(queue_times)

