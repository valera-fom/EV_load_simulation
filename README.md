# EV Load Forecasting Simulation

A comprehensive Streamlit application for simulating Electric Vehicle (EV) charging load forecasting with advanced Time of Use (ToU) optimization and multiple charging strategies.

## Features

### 🚗 EV Simulation
- **Dynamic EV Fleet Modeling** - Simulate realistic EV populations with varying battery capacities and charging rates
- **Multiple EV Models** - Support for different EV types with configurable parameters
- **Charging Infrastructure** - Various charger types and power levels

### ⚡ Time of Use (ToU) Optimization
- **Interactive Timeline** - 24-hour period assignment with visual color coding
- **Multiple Period Types** - Super Off-Peak, Off-Peak, Mid-Peak, and Peak periods
- **Smart Car Distribution** - Automatic proportional splitting of EVs among multiple periods of the same type
- **Real-time Adoption Rates** - Configurable adoption percentages for each period
- **Cycling Buttons** - Quick period assignment with visual feedback

### 🔋 Advanced Charging Strategies
- **PV + Battery System** - Solar charging during day, battery discharge during evening
- **Grid Battery Charging** - Off-peak grid charging with peak discharge
- **Vehicle-to-Grid (V2G)** - Bidirectional charging with discharge and recharge cycles
- **Smart Charging** - Time-based optimization for grid load balancing

### 📊 Comprehensive Analytics
- **Load Profile Visualization** - Real-time charging demand curves
- **Grid Impact Analysis** - Capacity utilization and overload detection
- **Multi-day Simulation** - Extended duration analysis with realistic patterns
- **Performance Metrics** - Detailed statistics and optimization results

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/valera-fom/EV_load_simulation.git
   cd EV_load_simulation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your grid data** (optional)
   - Place your grid data CSV files in the `datasets/` directory
   - Files should have: Date, Active power, Spare power columns
   - Or use the provided sample data

4. **Run the application**
   ```bash
   streamlit run pages/main_simulation.py
   ```

## Usage

### Basic Setup
1. **Configure EV Fleet** - Set total EVs, battery capacities, and charging rates
2. **Define Time of Use Periods** - Assign hours to different tariff periods using the interactive timeline
3. **Set Adoption Rates** - Configure percentage of users for each period
4. **Enable Strategies** - Activate PV, V2G, or other optimization strategies
5. **Run Simulation** - Execute and analyze results

### Advanced Features
- **Multiple Peak Management** - Handle complex charging patterns with multiple peaks
- **Real-time Optimization** - Dynamic load balancing based on grid capacity
- **Scenario Analysis** - Compare different adoption rates and strategies
- **Export Results** - Save simulation data for further analysis

## Project Structure

```
EV_simulation/
├── pages/
│   ├── main_simulation.py      # Main Streamlit application with ToU optimization
│   └── vehicle_sandbox.py      # Vehicle testing and configuration interface
├── datasets/                   # Grid profile and historical data (not in repo)
│   ├── df1.csv                # Grid data file 1
│   ├── df2.csv                # Grid data file 2
│   └── df3.csv                # Grid data file 3
├── EV.py                      # EV models and fleet configuration
├── charger.py                  # Charging infrastructure models
├── sim_setup.py               # Simulation configuration and setup
├── charging_procces.py        # Charging algorithm implementation
├── ev_factory.py              # EV fleet generation utilities
├── app.py                     # Streamlit app entry point
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Key Technologies

- **Streamlit** - Interactive web application framework
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization and plotting
- **Python 3.8+** - Core programming language

## Data Privacy

- **Grid data files** are excluded from the repository for privacy/security
- **Sample data** can be added to the `datasets/` directory for testing
- **Configuration files** are included for easy setup

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **E-REDES** - Portuguese distribution system operator for grid data
- **Streamlit Community** - For the excellent web framework
- **Open Source Community** - For the various Python libraries used

## Contact

For questions or support, please open an issue on GitHub or contact the development team.


 