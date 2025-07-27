import simpy
import numpy as np
from EV import EV, EV_MODELS
from charger import Charger, CHARGER_MODELS
from charging_procces import EVChargingProcess
from ev_factory import create_charger
import random
import os
from datetime import datetime, timedelta

def clear_console():
    """Clear console output to prevent memory buildup from old simulation logs."""
    os.system('cls' if os.name == 'nt' else 'clear')

class SimulationSetup:
    def __init__(self, ev_counts, charger_counts, sim_duration=24*60, time_step=1, 
                 arrival_time_mean=12*60, arrival_time_span=4*60, grid_power_limit=0, verbose=False, silent=False):
        """
        ev_counts: dict of {ev_type: number_of_evs}
        charger_counts: dict of {charger_type: number_of_chargers}
        sim_duration: total simulation time in minutes (default 24h)
        time_step: time step in minutes (default 1 min)
        arrival_time_mean: mean arrival time in minutes (default 12h = noon)
        arrival_time_span: 2*sigma for normal distribution in minutes (default 4h)
        grid_power_limit: maximum grid power in kW (0 = no limit)
        verbose: enable detailed debug output (default False)
        silent: completely suppress all output (default False)
        """
        self.ev_counts = ev_counts
        self.charger_counts = charger_counts
        self.sim_duration = sim_duration
        self.time_step = time_step
        self.arrival_time_mean = arrival_time_mean
        self.arrival_time_span = arrival_time_span
        self.verbose = verbose
        self.silent = silent
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
        
        # Progress tracking for pretty logging
        self.total_evs = sum(ev_counts.values())
        self.evs_processed = set()  # Use set to avoid duplicates
        self.last_progress_time = 0
        # Show progress for all simulations
        self.progress_interval = max(1, self.total_evs // 10) if self.total_evs > 0 else 0
        
        self._init_chargers()
        self._init_evs()
        self.env.process(self.charging_coordinator())
        self._start_charger_processes()

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
        
        # Assign arrival times to EVs
        for i, ev in enumerate(self.evs):
            ev.arrival_time = max(0, arrival_times[i])  # Ensure non-negative arrival times

    def _start_charger_processes(self):
        """Start all charger processes."""
        for charger_type, charger_list in self.chargers.items():
            for charger in charger_list:
                process = self.env.process(self._charger_process(charger, charger_type))
                charger._process = process  # Store process reference for interruption

    def _charger_process(self, charger_instance, charger_type):
        """Individual charger process that handles EV charging."""
        while True:
            try:
                # Wait for a compatible EV in the global queue
                ev = yield self.global_ev_queue.get(lambda ev: self._is_compatible(ev, charger_type))
                
                # Check if this EV has already been processed
                if ev.name in self.evs_processed:
                    continue
                
                # Update progress counter
                self.evs_processed.add(ev.name) # Add to set
                
                # Show progress (only if not silent and at intervals)
                if not self.silent and self.progress_interval > 0 and len(self.evs_processed) % self.progress_interval == 0:
                    progress = (len(self.evs_processed) / self.total_evs) * 100
                    print(f"⚡ Progress: {len(self.evs_processed)}/{self.total_evs} EVs processed ({progress:.1f}%)")
                    
                    # Add warning if we're processing more EVs than expected
                    if len(self.evs_processed) > self.total_evs:
                        print(f"⚠️ Warning: Processing more EVs ({len(self.evs_processed)}) than expected ({self.total_evs})")
                
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
                    
            except simpy.Interrupt:
                # Simulation ended, stop this charger process
                break

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
        """Schedule all EVs with arrival times."""
        if not self.silent:
            print(f"\n🚗 Scheduling {len(self.evs)} EVs...")
            print(f"⏰ Arrival time: {self.arrival_time_mean/60:.1f}h ± {self.arrival_time_span/60:.1f}h")
            if self.grid_power_limit is not None:
                print(f"⚡ Grid power limit: {self.grid_power_limit} kW")
            elif self.grid_power_profile is not None:
                print(f"📊 Grid power profile: {len(self.grid_power_profile)} time steps")
            else:
                print(f"🔌 No grid power limit")
        
        # Generate arrival times
        self._generate_arrival_times()
        
        # Schedule each EV
        for ev in self.evs:
            self._ev_arrival(ev, ev.arrival_time)
        
        if not self.silent:
            print(f"✅ All {len(self.evs)} EVs scheduled")

    def _ev_arrival(self, ev, arrival_time):
        """Process EV arrival at the specified time."""
        yield self.env.timeout(arrival_time)
        
        # Add EV to global queue
        self.global_ev_queue.put(ev)
        ev.queued_at = self.env.now

    def reset_simulation(self):
        """Reset simulation state for clean runs."""
        self.evs_processed.clear()
        self.load_curve = np.zeros(self.sim_duration // self.time_step)
        self.current_grid_power = 0
        self.charging_pool.clear()
        self.rejected_evs.clear()
        self.global_ev_queue = simpy.FilterStore(self.env)

    def run_simulation(self):
        """
        Run the complete simulation.
        """
        # Reset simulation state for clean run
        self.reset_simulation()
        
        # Clear console to prevent memory buildup from old logs
        # Only clear if not verbose and not RL training and not Bayesian optimization
        if not self.verbose and not hasattr(self, '_rl_training') and not hasattr(self, '_bayesian_optimization'):
            clear_console()
        
        # Show initialization info
        if not self.silent:
            print(f"🔧 Initializing simulation...")
            ev_types = list(self.ev_counts.keys())
            charger_types = list(self.charger_counts.keys())
            print(f"📊 Data: {self.total_evs} EVs ({', '.join(ev_types)}), {sum(len(charger_list) for charger_list in self.chargers.values())} chargers ({', '.join(charger_types)}), {self.sim_duration/60:.1f}h duration")
        
        # Schedule EVs with arrival times
        self.schedule_evs()
        
        # Run the simulation
        if not self.silent:
            print(f"\n🔄 Starting simulation for {self.sim_duration/60:.1f} hours...")
            start_time = datetime.now()
        
        self.env.run(until=self.sim_duration)
        
        # Interrupt all charger processes to prevent over-counting
        for charger_type, charger_list in self.chargers.items():
            for charger in charger_list:
                if hasattr(charger, '_process') and charger._process is not None:
                    charger._process.interrupt()
        
        if not self.silent:
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"✅ Simulation completed in {duration.total_seconds():.1f}s")
            print(f"⏰ Simulated time: {self.env.now/60:.1f} hours")
        
        # Display rejection notifications
        if not self.silent:
            self._display_rejection_notifications()
        
        # Display charger usage statistics
        if not self.silent:
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
            print(f"\n❌ Rejected EVs ({len(self.rejected_evs)}):")
            for ev, reason in self.rejected_evs:
                print(f"  • {ev.name}: {reason}")
        else:
            print("\n✅ All EVs scheduled successfully!")
    
    def _display_charger_statistics(self):
        """
        Display charger usage statistics.
        """
        print(f"\n📊 Simulation Summary:")
        
        # Charger statistics
        total_chargers = sum(len(charger_list) for charger_list in self.chargers.values())
        print(f"🔌 Total chargers: {total_chargers}")
        for charger_type, charger_list in self.chargers.items():
            print(f"  • {charger_type}: {len(charger_list)} chargers")
        
        # Queue statistics
        queue_times = [ev.queue_time for ev in self.evs if hasattr(ev, 'queue_time') and ev.queue_time > 0]
        avg_queue_time = np.mean(queue_times) if queue_times else 0
        max_queue_time = np.max(queue_times) if queue_times else 0
        
        print(f"⏱️  Queue statistics:")
        print(f"  • EVs that waited: {len(queue_times)}/{len(self.evs)} ({len(queue_times)/len(self.evs)*100:.1f}%)")
        print(f"  • Average wait time: {avg_queue_time:.1f} minutes")
        print(f"  • Maximum wait time: {max_queue_time:.1f} minutes")
        
        # Load curve statistics
        if hasattr(self, 'load_curve') and len(self.load_curve) > 0:
            peak_load = np.max(self.load_curve)
            avg_load = np.mean(self.load_curve)
            total_energy = np.sum(self.load_curve) / 60  # Convert to kWh
            print(f"⚡ Load statistics:")
            print(f"  • Peak load: {peak_load:.1f} kW")
            print(f"  • Average load: {avg_load:.1f} kW")
            print(f"  • Total energy: {total_energy:.1f} kWh")
        
        # Store for external access
        self.avg_queue_time = avg_queue_time
        self.num_queued_evs = len(queue_times)

