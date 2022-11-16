from model import *
import re

def load_switches(config: dict) -> "list[Switch]":
    return Switch.load_from_config(config)

def load_microservices(config: dict) -> "list[Microservice]":
    return Microservice.load_from_config(config)

def load_hosts(config: dict) -> "list[Host]":
    return Host.load_from_config(config)

def load_workflows(config: dict) -> "list[Workflow]":
    return Workflow.load_from_config(config)

def load_links(config: dict) -> "list[Link]":
    return Link.load_from_config(config)

def load_parameters(config: dict) -> "dict":
    return config['parameters']

def get_z_vars(sol: dict) -> list:
    return list(filter(lambda x: re.search("^z(_[^_]+){3}$", x), sol.keys()))

def get_x_vars(sol: dict) -> list:
    return list(filter(lambda x: re.search("^x_[^_]+$", x), sol.keys()))

def get_y_vars(sol: dict) -> list:
    return list(filter(lambda x: re.search("^y(_[^_]+){2}$", x), sol.keys()))

def get_cf_vars(sol: dict) -> list:
    return list(filter(lambda x: re.search("^cf(_[^_]+){3}$", x), sol.keys()))

def get_f_vars(sol: dict) -> list:
    return list(filter(lambda x: re.search("^f(_[^_]+){5}$", x), sol.keys()))

def get_fprime_vars(sol: dict) -> list:
    return list(filter(lambda x: re.search("^fprime(_[^_]+){5}$", x), sol.keys()))
