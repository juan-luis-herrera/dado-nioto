# DADO/NIoTO prototype

This repository contains the DADO (Distributed Application Deployment Application) artifact, as well as the NIoTO (Next-generation IoT Optimization) artifact, as presented in the Ph. D. thesis _DADO: framework for the deployment of distributed IoT applications in Edge-Fog-Cloud Environments_.

## Requirements for the use of DADO/NIoTO

- Python 3.7.5 (or higher).
- `pip` Python package manager.
- `pandas`, `matplotlib`, and `networkx` Python packages (installation instructions below).
- `gurobipy` Python package, for the version you have a license for (installation instructions below).
- Gurobi solver.

## Installing DADO's dependencies

### Windows

```bash
pip install -r requirements.txt
```

### Linux and macOS

```bash
pìp3 install -r requirements.txt
```

## Using DADO/NIoTO

The usage of DADO is as follows:

```bash
python condado.py -i <DADOJSON scenario> -obj <Objective QoS metric> -o <Solution CSV output> [-trep <Timing report output>] [-debug]
```

### Arguments

| Argument | Description | Format | Type |
| --- | --- | --- | --- |
| DADOJSON scenario | Configuration of the scenario from the previous time instant. | File path (\*.dadojson) | Mandatory |
| Objective QoS metric | QoS metric that the system should optimize. Possible metrics are `avg_response_time` (single-objective average response time, original DADO), `overall_cost` (single-objective deploymnent cost), `capex` (single-objective deployment CAPEX), `opex` (single-objective deployment OPEX), or `multi` (multi-objective cost and response time, NIoTO) | `{avg_response_time,overall_cost,capex,opex,multi}` | Mandatory |
| Solution CSV output | Deployment determined by DADO. | File path (\*.csv) | Mandatory |
| Timing report output | Report of time taken per step by DADO | File path (\*.csv) | Optional |
| `-debug` | Enable debug mode | N/A | Optional