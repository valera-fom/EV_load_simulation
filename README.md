# EV Load Simulation Tool

A comprehensive Streamlit application for simulating Electric Vehicle (EV) charging load patterns and their impact on electrical grid infrastructure. Features advanced optimization strategies including Time of Use (TOU) pricing, PV+battery systems, Vehicle-to-Grid (V2G), and grid battery storage.

## 🎯 Key Features

### 🚗 EV Simulation & Configuration
- **Dynamic EV Fleet Modeling** - Configure battery capacity, charging rates, and initial state of charge
- **Flexible Charging Infrastructure** - Set AC charger count and power levels
- **Time-based Arrival Patterns** - Configure multiple charging peaks with custom timing and duration
- **Realistic Charging Algorithms** - EV arrival/departure patterns with intelligent charging behavior

### ⚡ Time of Use (TOU) Optimization
- **Interactive TOU Configuration** - Set up 2-5 charging periods with visual timeline
- **Smart Charging Adoption** - Configurable adoption percentages for each period (total ≤ 100%)
- **Peak/Off-Peak Logic** - Optimize charging during lower-cost electricity periods
- **Automatic Validation** - Ensures proper period configuration and adoption rates

### 🔋 Advanced Optimization Strategies
- **Smart Charging** - Time-based optimization for grid load balancing
- **PV + Battery System** - Solar charging during day, battery discharge during evening peak
- **Grid Battery Storage** - Off-peak grid charging with peak discharge for load balancing
- **Vehicle-to-Grid (V2G)** - Bidirectional charging with discharge and recharge cycles

### 📊 Analysis & Optimization Tools
- **Capacity Analysis** - Find maximum EV count for given infrastructure constraints
- **Dynamic Load Optimizer** - AI-powered gradient-based optimization for TOU periods
- **Peak Demand Analysis** - Visualize grid load curves with EV charging overlay
- **Strategy Impact Comparison** - Compare effects of different optimization approaches

### 📈 Data & Visualization
- **Multiple Data Sources** - Upload Excel files or generate AI-powered synthetic load curves
- **Interactive Charts** - Real-time visualization of load curves and optimization effects
- **Professional PDF Reports** - Generate detailed analysis reports with customizable content
- **Scenario Management** - Save and load different simulation configurations
- **Custom Scenarios** - Create and save custom simulation scenarios with all parameters
- **Graph Controls** - Interactive controls for battery effects, legend, grid lines, and smoothing

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/valera-fom/EV_load_simulation.git
   cd EV_load_simulation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Basic Usage

1. **Data Setup** (Required First)
   - Upload Excel file with datetime and load data, OR
   - Switch to 'Synthetic Generation' and configure parameters, OR
   - Use 'Generate Synthetic Curve' button for quick start

2. **Configuration** (Sidebar)
   - Set EV battery capacity and charging rate
   - Configure AC charger count and power
   - Set simulation duration and charging peaks

3. **Time of Use Setup** (Optional)
   - Enable 'Smart Charging' in strategies
   - Configure peak/off-peak periods and adoption percentages
   - Ensure total adoption ≤ 100%

4. **Optimization Strategies** (Optional)
   - Enable PV + Battery, Grid Battery, or V2G strategies
   - Configure strategy-specific parameters

5. **Custom Scenarios** (Optional)
   - Create custom scenarios with all simulation parameters
   - Save scenarios to JSON files for future use
   - Load and apply custom scenarios from the dropdown

6. **Run Simulation**
   - Click 'Run Simulation' to execute analysis
   - Use graph controls to customize visualization
   - View results and generate PDF reports

## 📁 Project Structure

```
EV_simulation/
├── app.py                     # Main application entry point
├── pages/
│   ├── main_simulation.py     # Core simulation interface with TOU optimization
│   ├── vehicle_sandbox.py     # Vehicle testing and configuration interface
│   └── components/
│       ├── tou_optimizer.py   # TOU period optimization engine
│       ├── capacity_analyzer.py # Grid capacity analysis utilities
│       ├── gradient_optimizer_ui.py # Dynamic load optimization interface
│       ├── pdf_generator.py   # Professional PDF report generation
│       └── session_manager.py # Configuration save/load functionality
├── portable_load_generator.py # AI-powered synthetic load curve generation
├── portable_models/           # Trained models for synthetic data generation
│   ├── weekday/              # Weekday load patterns
│   └── weekend/              # Weekend load patterns
├── saved_scenarios/          # Custom scenarios saved as JSON files
├── EV.py                     # EV models and fleet configuration
├── charger.py                # Charging infrastructure models
├── sim_setup.py             # Simulation configuration and setup
├── charging_procces.py      # Charging algorithm implementation
├── ev_factory.py            # EV fleet generation utilities
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```

## 🔧 Technical Details

### Simulation Engine
- **EV Arrival/Departure Patterns** - Realistic charging behavior simulation
- **Charging Algorithms** - Intelligent load management based on grid capacity
- **Time Resolution** - 1-minute resolution simulation with 15-minute data upsampling
- **Multi-day Analysis** - Extended 48-hour simulation capabilities

### Data Processing
- **Excel File Import** - Support for datetime and load data columns using openpyxl
- **Synthetic Generation** - AI-powered load curve generation using trained models
- **Data Validation** - Automatic validation of uploaded data formats
- **Grid Profile Processing** - Capacity analysis and margin curve calculation

### Optimization Algorithms
- **Capacity Analysis** - Maximum EV count calculation for infrastructure constraints
- **Gradient-based Optimization** - Dynamic TOU period optimization
- **Strategy Simulation** - PV, V2G, and grid battery effect modeling
- **Peak Demand Reduction** - Intelligent load balancing algorithms

### Output & Visualization
- **Interactive Plots** - Real-time load curve visualization with optimization effects
- **Graph Controls** - Toggle battery effects, legend, grid lines, and smoothing
- **PDF Report Generation** - Professional analysis reports with customizable themes
- **Data Export** - Simulation results and configuration export
- **Session Management** - Save and restore simulation configurations
- **Custom Scenarios** - JSON-based scenario storage and loading

## 🎯 Use Cases

### Utility Companies
- **Grid Planning** - Assess infrastructure requirements for EV adoption
- **Load Forecasting** - Predict peak demand with EV charging patterns
- **TOU Strategy Development** - Optimize pricing periods for load balancing
- **Capacity Planning** - Determine optimal EV count for existing infrastructure

### Researchers & Analysts
- **EV Impact Studies** - Analyze grid effects of different EV adoption scenarios
- **Strategy Comparison** - Compare effectiveness of various optimization approaches
- **Data Analysis** - Process and visualize real grid data with EV charging
- **Scenario Testing** - Test different EV fleet compositions and charging patterns

### Energy Planners
- **Infrastructure Sizing** - Determine charging station requirements
- **Peak Demand Management** - Develop strategies for load balancing
- **Renewable Integration** - Analyze PV + battery system effectiveness
- **Grid Stability** - Assess V2G and grid battery impact on system stability

## 📊 Analysis Capabilities

- **Grid Load Curve Visualization** - EV charging overlay on base load
- **Peak Demand Analysis** - Optimization effects and capacity utilization
- **Time-of-Use Pattern Simulation** - Smart charging behavior modeling
- **Maximum EV Capacity Calculation** - Infrastructure constraint analysis
- **Strategy Impact Comparison** - TOU, PV, V2G, and grid battery effects
- **Custom Scenario Management** - Save and load complete simulation configurations

## 🎛️ Graph Controls

The application includes interactive graph controls for enhanced visualization:

- **🔋 Battery Effects** - Toggle display of PV battery, grid battery, and V2G effects
- **📋 Legend** - Show/hide graph legend
- **🔄 Smooth Lines** - Switch between smooth and stepped line visualization
- **🛡️ Safety Margin** - Display 20% safety margin line
- **📐 Grid Lines** - Show/hide grid lines on graphs
- **📊 Average Line** - Display horizontal average line on capacity graph
- **🔄 Refresh Graph** - Manually refresh graph display

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **E-REDES** - Portuguese distribution system operator for grid data inspiration
- **Streamlit Community** - Excellent web application framework
- **Open Source Community** - Various Python libraries and tools

## 📞 Contact

For questions, support, or contributions, please open an issue on GitHub or contact the development team.

**Email:** valerii.fomenko@ukr.net


 