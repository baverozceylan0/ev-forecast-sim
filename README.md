# ChargeCast: Forecasting EV Charging Demand with Integrated Simulation &amp; Scheduling

**ChargeCast** is a modular machine learning pipeline for forecasting electric vehicle (EV) charging demand in large-scale charging facilities. It integrates **forecasting**, **simulation**, and **scheduling** to help researchers and operators evaluate and optimize EV charging strategies.

> **Status:** 🚧 *Ongoing project* – This repository is under active development. Many features are incomplete, and there are open issues and bugs. Contributions and feedback are welcome.

## Installation
```bash
# Clone the repository
git clone https://github.com/baverozceylan0/ev-forecast-sim.git
cd ev-forecast-sim

# Start containers
docker-compose up -d

# Open a shell inside the container
docker exec -it ev-forecast-sim bash

# Run the pipeline (example)
python main.py
