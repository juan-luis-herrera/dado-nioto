class Switch:

    SWITCHES_TAG = "switches"
    ID_TAG = "id"
    CAPEX_TAG = "capex"
    OPEX_TAG = "opex"
    CONTROLLER_CAPEX_TAG = "capex_cnt"
    CONTROLLER_OPEX_TAG = "opex_cnt"
    MIGRATION_COST_CNT_TAG = "migration_cost_cnt"

    def __init__(self, sw_dct: dict):
        self.id = sw_dct[self.ID_TAG]
        self.capex = sw_dct.get(self.CAPEX_TAG, 0)
        self.opex = sw_dct.get(self.OPEX_TAG, 0)
        self.capex_cnt = sw_dct.get(self.CONTROLLER_CAPEX_TAG, 0)
        self.opex_cnt = sw_dct.get(self.CONTROLLER_OPEX_TAG, 0)
        self.migration_cost_cnt = sw_dct.get(self.MIGRATION_COST_CNT_TAG, 0)
    
    @classmethod
    def load_from_config(cls, config: dict) -> "list[Switch]":
        switches = []
        cnf_sw = config[cls.SWITCHES_TAG]
        for switch in cnf_sw:
            switches.append(Switch(switch))
        return switches

    def to_dict(self) -> dict:
        return {self.ID_TAG: self.id, self.CAPEX_TAG: self.capex,
            self.OPEX_TAG: self.opex, self.CONTROLLER_CAPEX_TAG: self.capex_cnt,
            self.CONTROLLER_OPEX_TAG: self.opex_cnt, 
            self.MIGRATION_COST_CNT_TAG: self.migration_cost_cnt}
    
    def __str__(self):
        return "<Switch: " + str(self.to_dict()) +">"
    
    def __repr__(self):
        return self.__str__()

    
class Microservice:

    MICROSERVICES_TAG = "microservices"
    ID_TAG = "id"
    CYCLES_TAG = "cycles"
    INPUT_TAG = "input"
    OUTPUT_TAG = "output"
    MEMORY_TAG = "memory"
    MIGRATION_COST_TAG = "migration_cost"

    def __init__(self, mi_dct: dict):
        self.id = mi_dct[self.ID_TAG]
        self.cycles = mi_dct[self.CYCLES_TAG]
        self.input = mi_dct[self.INPUT_TAG]
        self.output = mi_dct[self.OUTPUT_TAG]
        self.memory = mi_dct[self.MEMORY_TAG]
        self.migration_cost = mi_dct.get(self.MIGRATION_COST_TAG, 0)
    
    @classmethod
    def load_from_config(cls, config: dict) -> "list[Microservice]":
        microservices = []
        cnf_mi = config[cls.MICROSERVICES_TAG]
        for microservice in cnf_mi:
            microservices.append(Microservice(microservice))
        return microservices
    
    def to_dict(self) -> dict:
        return {self.ID_TAG: self.id, self.CYCLES_TAG: self.cycles,
            self.INPUT_TAG: self.input, self.OUTPUT_TAG: self.output,
            self.MEMORY_TAG: self.memory, self.MIGRATION_COST_TAG: self.migration_cost}
    
    def __str__(self):
        return "<Microservice: " + str(self.to_dict()) +">"

    def __repr__(self):
        return self.__str__()

class Host:

    HOSTS_TAG = "hosts"
    ID_TAG = "id"
    POWER_TAG = "power"
    MEMORY_TAG = "memory"
    CAPEX_TAG = "capex"
    CYCLE_OPEX_TAG = "opex_cycle"
    MEMORY_OPEX_TAG = "opex_memory"

    def __init__(self, h_dct: dict):
        self.id = h_dct[self.ID_TAG]
        self.power = h_dct[self.POWER_TAG]
        self.memory = h_dct[self.MEMORY_TAG]
        self.capex = h_dct.get(self.CAPEX_TAG, 0)
        self.opex_cycle = h_dct.get(self.CYCLE_OPEX_TAG, 0)
        self.opex_memory = h_dct.get(self.MEMORY_OPEX_TAG, 0)

    @classmethod
    def load_from_config(cls, config: dict) -> "list[Host]":
        hosts = []
        cnf_h = config[cls.HOSTS_TAG]
        for host in cnf_h:
            hosts.append(Host(host))
        return hosts
    
    def to_dict(self) -> dict:
        return {self.ID_TAG: self.id, self.POWER_TAG: self.power,
                self.MEMORY_TAG: self.memory, self.CAPEX_TAG: self.capex,
                self.CYCLE_OPEX_TAG: self.opex_cycle, self.MEMORY_OPEX_TAG: self.opex_memory}

    def __str__(self):
        return "<Host: " + str(self.to_dict()) + ">"

    def __repr__(self):
        return self.__str__()

class Workflow:

    WORKFLOWS_TAG = "workflows"
    ID_TAG = "id"
    CHAIN_TAG = "chain"
    STARTER_TAG = "starter"
    RESPONSE_TAG = "response"
    MAX_RESPONSE_TIME_TAG = "max_response_time"

    def __init__(self, wf_dct: dict):
        self.id = wf_dct[self.ID_TAG]
        self.chain = wf_dct[self.CHAIN_TAG]
        self.starter = wf_dct[self.STARTER_TAG]
        self.response = wf_dct[self.RESPONSE_TAG]
        self.max_response_time = wf_dct.get(self.MAX_RESPONSE_TIME_TAG, 0)

    @classmethod
    def load_from_config(cls, config: dict) -> "list[Workflow]":
        workflows = []
        cnf_wf = config[cls.WORKFLOWS_TAG]
        for workflow in cnf_wf:
            workflows.append(Workflow(workflow))
        return workflows
    
    def to_dict(self) -> dict:
        return {self.ID_TAG: self.id, self.CHAIN_TAG: self.chain,
                self.STARTER_TAG: self.starter, self.RESPONSE_TAG: self.response,
                self.MAX_RESPONSE_TIME_TAG: self.max_response_time}
            
    def is_starter(self, host: Host) -> int:
        return 1 if host.id == self.starter else 0
    
    def response_type(self) -> int:
        return 1 if self.response else 0

    def __str__(self):
        return "<Workflow: " + str(self.to_dict()) + ">"

    def __repr__(self):
        return self.__str__()


class Link:

    LINKS_TAG = "links"
    SOURCE_TAG = "source"
    DESTINATION_TAG = "destination"
    LATENCY_TAG = "latency"
    CAPACITY_TAG = "capacity"
    CAPEX_TAG = "capex"
    OPEX_TAG = "opex"

    def __init__(self, l_dct: dict):
        self.source = l_dct[self.SOURCE_TAG]
        self.destination = l_dct[self.DESTINATION_TAG]
        self.latency = abs(l_dct[self.LATENCY_TAG])
        self.capacity = l_dct[self.CAPACITY_TAG]
        self.capex = l_dct.get(self.CAPEX_TAG, 0)
        self.opex = l_dct.get(self.OPEX_TAG, 0)

    @classmethod
    def load_from_config(cls, config: dict, custom_dir: bool = False) -> "list[Link]":
        links = []
        cnf_l = config[cls.LINKS_TAG]
        sources_dests = []
        for link in cnf_l:
            if custom_dir:
                links.append(Link(link))
            else:
                lnk_id = (link[cls.SOURCE_TAG], link[cls.DESTINATION_TAG])
                if lnk_id not in sources_dests:
                    sources_dests.append(lnk_id)
                    links.append(Link(link))
                    rev_id = (lnk_id[1], lnk_id[0])
                    if rev_id not in sources_dests:
                        sources_dests.append(rev_id)
                        rev = link.copy()
                        rev[cls.SOURCE_TAG] = rev_id[0]
                        rev[cls.DESTINATION_TAG] = rev_id[1]
                        links.append(Link(rev))
        return links
    
    def to_dict(self) -> dict:
        return {self.SOURCE_TAG: self.source, self.DESTINATION_TAG: self.destination,
                self.LATENCY_TAG: abs(self.latency), self.CAPACITY_TAG: self.capacity,
                self.CAPEX_TAG: self.capex, self.OPEX_TAG: self.opex}

    def __str__(self):
        return "<Link: " + str(self.to_dict()) + ">"

    def __repr__(self):
        return self.__str__()
