import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from concurrent.futures import ThreadPoolExecutor, Future, wait
from multiprocessing import cpu_count
from model import *
from time import perf_counter
import unmarshalling
import argparse
from math import inf

class ParallelDADO:

    PARAMETERS_TAG = "parameters"
    CONTROLLERS_TAG = "controllers"
    CONTROL_SIZE_TAG = "control_size"
    CAPEX_BUDGET_TAG = "capex_budget"
    OPEX_BUDGET_TAG = "opex_budget"

    def __init__(self, config: dict):

        self.__times = {"startup": perf_counter()}
        self.__config = config
        self.__executor = ThreadPoolExecutor(cpu_count())
        self.__futures = []

        # Loading elements
        self.__switches = unmarshalling.load_switches(config)
        self.__microservices = unmarshalling.load_microservices(config)
        self.__hosts = unmarshalling.load_hosts(config)
        self.__workflows = unmarshalling.load_workflows(config)
        self.__links = unmarshalling.load_links(config)
        self.__parameters = self.load_parameters(config)
        self.__times['loading'] = perf_counter()

        # Creating dictionaries of elements for convenience
        self.__dkt_switches = self.__dict_by_id(self.__switches)
        self.__dkt_microservices = self.__dict_by_id(self.__microservices)
        self.__dkt_hosts = self.__dict_by_id(self.__hosts)
        self.__dkt_workflows = self.__dict_by_id(self.__workflows)
        self.__dkt_links = self.__dict_links(self.__links)

        # Initializing elements for convenience
        self.__capex = None
        self.__opex = None

        # Quick feasibility checks
        self.__basic_feas_check()

        self.__mip_model = gp.Model('DADO')

        # Creating the MIP variables
        self.__u = None
        var_creation = [self.__executor.submit(self.__create_z_variables), self.__executor.submit(self.__create_f_variables), self.__executor.submit(self.__create_fprime_variables),
                        self.__executor.submit(self.__create_x_variables), self.__executor.submit(
                            self.__create_y_variables),
                        self.__executor.submit(self.__create_cf_variables)]
        wait(var_creation)
        self.__mip_model.update()
        self.__times['variable_creation'] = perf_counter()

        # Creating the MIP constraints
        self.__futures += self.__add_single_microservice_constraint()
        self.__futures += self.__add_memory_constraint()
        self.__futures += self.__add_controller_mapping_constraint()
        self.__futures += self.__add_active_controller_constraint()
        self.__futures += self.__add_flow_constraints()
        self.__futures += self.__add_ctrl_flow_constraints()
        self.__futures += self.__add_link_capacity_constraint()
        self.__futures += self.__add_max_rt_constraint()
        if self.__config.get(self.CAPEX_BUDGET_TAG):
            self.__u = self.__usage_vars()
            self.__capex = self.__calc_capex()
            self.__add_capex_budget(self.__config[self.CAPEX_BUDGET_TAG])
        if self.__config.get(self.OPEX_BUDGET_TAG):
            if self.__u is None:
                self.__u = self.__usage_vars()
            self.__opex = self.__calc_opex()
            self.__add_opex_budget(self.__config[self.OPEX_BUDGET_TAG])

        self.__times['constraint_creation'] = perf_counter()

        self.__optimized = False
        self.__exportable = False
        self.__suboptimal = False

    @classmethod
    def load_parameters(cls, config: dict) -> dict:
        return config[cls.PARAMETERS_TAG]
    
    @staticmethod
    def __dict_by_id(elements_with_id: list) -> dict:
        return {element.id : element for element in elements_with_id}
    
    @staticmethod
    def __dict_links(links: list) -> dict:
        return {(link.source, link.destination) : link for link in links}
    
    def __basic_feas_check(self) -> None:
        total_mem = sum([h.memory for h in self.__hosts])
        total_req = sum([self.__dkt_microservices[m].memory for wf in self.__workflows for m in wf.chain])
        if total_mem > total_req:
            print('Initial feasibility check passed')
        else:
            raise RuntimeError('Feasibility check failed. Total memory: {}. Required memory: {}'.format(total_mem, total_req))
    
    def __create_z_variables(self) -> None:
        self.__z = {
            host.id : {
                workflow.id : {
                    microservice: self.__mip_model.addVar(name='z_{}_{}_{}'.format(
                        host.id, workflow.id, microservice), vtype=GRB.BINARY)
                    for microservice in workflow.chain
                }
                for workflow in self.__workflows
            }
            for host in self.__hosts
        }
    
    def __create_f_variables(self) -> None:
        self.__f = {
            (link.source, link.destination) : {
                host.id : {
                    workflow.id : {
                        microservice : self.__mip_model.addVar(name='f_{}_{}_{}_{}_{}'.format(
                            link.source, link.destination, host.id, workflow.id, microservice)
                        , vtype=GRB.BINARY)
                        for microservice in workflow.chain
                    }
                    for workflow in self.__workflows
                }
                for host in self.__hosts
            }
            for link in self.__links
        }
    
    def __create_fprime_variables(self) -> None:
        self.__fprime = {
            (link.source, link.destination) : {
                host.id : {
                    workflow.id : self.__mip_model.addVar(name='fprime_{}_{}_{}_{}'.format(
                        link.source, link.destination, host.id, workflow.id)
                    , vtype=GRB.BINARY)
                    for workflow in self.__workflows
                }
                for host in self.__hosts
            }
            for link in self.__links
        }
    
    def __create_x_variables(self) -> None:
        self.__x = {
            switch.id : self.__mip_model.addVar(name='x_{}'.format(switch.id), vtype=GRB.BINARY)
            for switch in self.__switches
        }
    
    def __create_y_variables(self) -> None:
        self.__y = {
            switch.id : {
                controller.id : self.__mip_model.addVar(name='y_{}_{}'.format(switch.id, controller.id), vtype=GRB.BINARY)
                for controller in self.__switches
            }
            for switch in self.__switches
        }
    
    def __create_cf_variables(self) -> None:
        self.__cf = {
            (link.source, link.destination) : {
                switch.id : self.__mip_model.addVar(name='cf_{}_{}_{}'.format(
                    link.source, link.destination, switch.id)
                , vtype=GRB.BINARY)
                for switch in self.__switches
            }
            for link in self.__links
        }
    
    def __inner_ms_constraint(self, t: Workflow, m: str) -> None:
        self.__mip_model.addConstr(gp.quicksum(self.__z[h.id][t.id][m] for h in self.__hosts)== 1, 'max_mapping_{}_{}'.format(t.id, m))
        
    def __add_single_microservice_constraint(self) -> "list[Future]":
            return [self.__executor.submit(self.__inner_ms_constraint, t=t, m=m) for t in self.__workflows for m in t.chain]
    
    def __inner_mem_constraint(self, h: Host) -> None:
        self.__mip_model.addConstr(gp.quicksum(self.__z[h.id][t.id][m]*self.__dkt_microservices[m].memory for t in self.__workflows for m in t.chain) <= h.memory, 'max_power_{}'.format(h.id))
        
    def __add_memory_constraint(self) -> "list[Future]":
            return [self.__executor.submit(self.__inner_mem_constraint, h=h) for h in self.__hosts]
    
    """ def __add_single_thread_constraint(self) -> None:
        for h in self.__hosts:
            for t in self.__workflows:
                for m in t.chain:
                    self.__mip_model.add_constr(self.__z[h.id][t.id][m]*self.__dkt_microservices[m].cycles
                    <= h.power, 'single_core_{}_{}_{}'.format(h.id, t.id, m)) """
    
    def __inner_max_ctrl_constraint(self) -> None:
        if self.CONTROLLERS_TAG in self.__parameters:
            self.__mip_model.addConstr(gp.quicksum(self.__x[s.id] for s in self.__switches)
                                           <= self.__parameters[self.CONTROLLERS_TAG], 'maximum_controllers')

    def __add_maximum_controllers_constraint(self) -> "list[Future]":
        return [self.__executor.submit(self.__inner_max_ctrl_constraint)]
    
    def __inner_ctrl_mapping_constraint(self, s: Switch) -> None:
        self.__mip_model.addConstr(gp.quicksum(
            self.__y[s.id][c.id] for c in self.__switches) == 1, 'ctrl_mapping_{}'.format(s.id))
        
    def __add_controller_mapping_constraint(self) -> "list[Future]":
        return [self.__executor.submit(self.__inner_ctrl_mapping_constraint, s=s) for s in self.__switches]
    
    def __inner_active_controller_constraint(self, s: Switch, c: Switch) -> None:
        self.__mip_model.addConstr(
            self.__y[s.id][c.id] <= self.__x[c.id], 'active_controller_{}_{}'.format(s.id, c.id))

    def __add_active_controller_constraint(self) -> "list[Future]":
        return [self.__executor.submit(self.__inner_active_controller_constraint, s=s, c=c) for s in self.__switches for c in self.__switches]
    
    def __inner_starter_flow_constraint(self, t: Workflow) -> None:
        m0 = t.chain[0]
        for i in self.__hosts + self.__switches:
            j_adj = list(map(lambda l: l.destination, filter(
                lambda ll: ll.source == i.id, self.__links)))
            for h in self.__hosts:
                flow_linexp = gp.quicksum(self.__f[(
                    i.id, j)][h.id][t.id][m0] - self.__f[(j, i.id)][h.id][t.id][m0] for j in j_adj)
                if i in self.__switches:
                    self.__mip_model.addConstr(
                        flow_linexp == 0, 'flow_{}_{}_{}_{}'.format(i.id, h.id, t.id, m0))
                else:
                    if i.id == h.id:
                        self.__mip_model.addConstr(flow_linexp == t.is_starter(
                            h)*(1-self.__z[i.id][t.id][m0]), 'flow_{}_{}_{}_{}'.format(i.id, h.id, t.id, m0))
                    else:
                        self.__mip_model.addConstr(flow_linexp == -1*t.is_starter(
                            h)*self.__z[i.id][t.id][m0], 'flow_{}_{}_{}_{}'.format(i.id, h.id, t.id, m0))

    def __inner_returnal_flow_constraint(self, t: Workflow) -> None:
        if t.response:
            mn = t.chain[-1]
            for i in self.__hosts + self.__switches:
                j_adj = list(map(lambda l: l.destination, filter(
                    lambda ll: ll.source == i.id, self.__links)))
                for h in self.__hosts:
                    flow_prime_linexp = gp.quicksum(self.__fprime[(
                        i.id, j)][h.id][t.id] - self.__fprime[(j, i.id)][h.id][t.id] for j in j_adj)
                    if i in self.__switches:
                        self.__mip_model.addConstr(
                            flow_prime_linexp == 0, 'flow_{}_{}_{}_{}'.format(i.id, h.id, t.id, mn))
                    else:
                        if i.id == h.id:
                            self.__mip_model.addConstr(flow_prime_linexp == self.__z[h.id][t.id][mn]*(
                                1-t.is_starter(i))*t.response_type(), 'flow_prime_{}_{}_{}'.format(i.id, h.id, t.id))
                        else:
                            self.__mip_model.addConstr(flow_prime_linexp == -1*self.__z[h.id][t.id][mn]*t.is_starter(
                                i)*t.response_type(), 'flow_prime_{}_{}_{}'.format(i.id, h.id, t.id))

    def __inner_generic_flow_constraint(self, t: Workflow, m: str, m_minus: str) -> None:
        for i in self.__hosts + self.__switches:
            j_adj = list(map(lambda l: l.destination, filter(
                lambda ll: ll.source == i.id, self.__links)))
            for h in self.__hosts:
                flow_gen_linexp = gp.quicksum(self.__f[(
                    i.id, j)][h.id][t.id][m] - self.__f[(j, i.id)][h.id][t.id][m] for j in j_adj)
                if i in self.__switches:
                    self.__mip_model.addConstr(
                        flow_gen_linexp == 0, 'flow_{}_{}_{}_{}'.format(i.id, h.id, t.id, m))
                else:
                    if i.id == h.id:
                        cheat_z_var_1 = self.__mip_model.addVar(
                            name='z_prime_{}_{}_{}_{}'.format(h.id, i.id, t.id, m), vtype=GRB.BINARY)
                        self.__mip_model.addConstr(
                            -1*self.__z[h.id][t.id][m_minus]+cheat_z_var_1 <= 0)
                        self.__mip_model.addConstr(
                            -1+self.__z[i.id][t.id][m]+cheat_z_var_1 <= 0)
                        self.__mip_model.addConstr(
                            self.__z[h.id][t.id][m_minus]+1-self.__z[i.id][t.id][m]-cheat_z_var_1 <= 1)
                        self.__mip_model.addConstr(
                            flow_gen_linexp == cheat_z_var_1, 'flow_{}_{}_{}_{}'.format(i.id, h.id, t.id, m))
                    else:
                        cheat_z_var_2 = self.__mip_model.addVar(
                            name='z_double_{}_{}_{}_{}'.format(h.id, i.id, t.id, m), vtype=GRB.BINARY)
                        self.__mip_model.addConstr(
                            -1*self.__z[h.id][t.id][m_minus]+cheat_z_var_2 <= 0)
                        self.__mip_model.addConstr(
                            -1*self.__z[i.id][t.id][m]+cheat_z_var_2 <= 0)
                        self.__mip_model.addConstr(
                            self.__z[h.id][t.id][m_minus]+self.__z[i.id][t.id][m]-cheat_z_var_2 <= 1)
                        self.__mip_model.addConstr(
                            flow_gen_linexp == -1*cheat_z_var_2, 'flow_{}_{}_{}_{}'.format(i.id, h.id, t.id, m))

    def __add_flow_constraints(self) -> "list[Future]":
        all_flow_constrs = []
        # Starter flows
        for t in self.__workflows:
            all_flow_constrs.append(self.__executor.submit(
                self.__inner_starter_flow_constraint, t=t))
            all_flow_constrs.append(self.__executor.submit(
                self.__inner_returnal_flow_constraint, t=t))
            for m, m_minus in zip(t.chain[1:], t.chain):
                all_flow_constrs.append(self.__executor.submit(
                    self.__inner_generic_flow_constraint, t=t, m=m, m_minus=m_minus))
        return all_flow_constrs

    def __inner_ctrl_flow_constraint(self, s: Switch) -> None:
        if len(self.__rich['cf'].query("Switch == @s.id and Value > 0")) == 0:
            for i in self.__hosts + self.__switches:
                j_adj = list(map(lambda l: l.destination, filter(
                    lambda ll: ll.source == i.id, self.__links)))
                ctrl_flow_linexp = gp.quicksum(
                    self.__cf[(i.id, j)][s.id] - self.__cf[(j, i.id)][s.id] for j in j_adj)
                if i in self.__hosts:
                    self.__mip_model.addConstr(
                        ctrl_flow_linexp == 0, 'ctrl_flow_{}_{}'.format(i.id, s.id))
                else:
                    if i.id == s.id:
                        self.__mip_model.addConstr(
                            ctrl_flow_linexp == 1 - self.__y[s.id][i.id], 'ctrl_flow_{}_{}'.format(i.id, s.id))
                    else:
                        self.__mip_model.addConstr(
                            ctrl_flow_linexp == -1*self.__y[s.id][i.id], 'ctrl_flow_{}_{}'.format(i.id, s.id))

    def __add_ctrl_flow_constraints(self) -> "list[Future]":
        return [self.__executor.submit(self.__inner_ctrl_flow_constraint, s=s) for s in self.__switches]
    
    def __inner_link_capacity_constraint(self, l: Link) -> None:
        total_sum = 0
        for t in self.__workflows:
            for h in self.__hosts:
                total_sum += gp.quicksum(self.__f[(l.source, l.destination)][h.id]
                                         [t.id][m] * self.__dkt_microservices[m].input for m in t.chain)
                total_sum += self.__fprime.get((l.source, l.destination), {}).get(
                    h.id, {}).get(t.id, 0) * self.__dkt_microservices[t.chain[-1]].output
        total_sum += gp.quicksum(self.__cf[(l.source, l.destination)][s.id]
                                 * self.__parameters[self.CONTROL_SIZE_TAG] for s in self.__switches)
        self.__mip_model.addConstr(
            total_sum <= l.capacity, 'capacity_{}_{}'.format(l.source, l.destination))

    def __add_link_capacity_constraint(self) -> "list[Future]":
        return [self.__executor.submit(self.__inner_link_capacity_constraint, l=l) for l in self.__links]

    def __inner_link_usage(self, l: Link) -> gp.Var:
        my_var = self.__mip_model.addVar(name='u_l_{}_{}'.format(
            l.source, l.destination), vtype=GRB.BINARY)
        for h_id in self.__f[(l.source, l.destination)]:
            for t_id in self.__f[(l.source, l.destination)][h_id]:
                for m in self.__f[(l.source, l.destination)][h_id][t_id]:
                    self.__mip_model.addConstr(
                        my_var >= self.__f[(l.source, l.destination)][h_id][t_id][m])
                self.__mip_model.addConstr(
                    my_var >= self.__fprime[(l.source, l.destination)][h_id][t_id])
        for s_id in self.__cf[(l.source, l.destination)]:
            self.__mip_model.addConstr(
                my_var >= self.__cf[(l.source, l.destination)][s_id])
        return my_var

    def __inner_switch_usage(self, s: Switch) -> gp.Var:
        my_var = self.__mip_model.addVar(
            name='u_s_{}'.format(s.id), vtype=GRB.BINARY)
        for l in list(filter(lambda x: x.source == s.id, self.__links)):
            for h_id in self.__f[(l.source, l.destination)]:
                for t_id in self.__f[(l.source, l.destination)][h_id]:
                    for m in self.__f[(l.source, l.destination)][h_id][t_id]:
                        self.__mip_model.addConstr(
                            my_var >= self.__f[(l.source, l.destination)][h_id][t_id][m])
                    self.__mip_model.addConstr(
                        my_var >= self.__fprime[(l.source, l.destination)][h_id][t_id])
            for t_id in self.__cf[(l.source, l.destination)]:
                self.__mip_model.addConstr(
                    my_var >= self.__cf[(l.source, l.destination)][t_id])
        return my_var

    def __inner_host_usage(self, h: Host) -> gp.Var:
        my_var = self.__mip_model.addVar(
            name='u_h_{}'.format(h.id), vtype=GRB.BINARY)
        for t_id in self.__z[h.id]:
            for m in self.__z[h.id][t_id]:
                self.__mip_model.addConstr(
                    my_var >= self.__z[h.id][t_id][m])
        return my_var

    def __inner_cycles_usage(self, h: Host) -> gp.LinExpr:
        return gp.quicksum(self.__z[h.id][t.id][m]*self.__dkt_microservices[m].cycles for t in self.__workflows for m in t.chain)

    def __inner_memory_usage(self, h: Host) -> gp.LinExpr:
        return gp.quicksum(self.__z[h.id][t.id][m]*self.__dkt_microservices[m].memory for t in self.__workflows for m in t.chain)

    def __usage_vars(self) -> dict:
        usage_vars = {"links": {}, "switches": {},
                      "hosts": {}, "cycles": {}, "memory": {}}
        for l in self.__links:
            usage_vars["links"][(l.source, l.destination)] = self.__executor.submit(
                self.__inner_link_usage, l)
        for s in self.__switches:
            usage_vars["switches"][s.id] = self.__executor.submit(
                self.__inner_switch_usage, s)
        for h in self.__hosts:
            usage_vars["hosts"][h.id] = self.__executor.submit(
                self.__inner_host_usage, h)
            usage_vars["cycles"][h.id] = self.__executor.submit(
                self.__inner_cycles_usage, h)
            usage_vars["memory"][h.id] = self.__executor.submit(
                self.__inner_memory_usage, h)
        u_vars = {k: {subk: usage_vars[k][subk].result()} for k in usage_vars for subk in usage_vars[k]}
        self.__mip_model.update()
        return u_vars
    
    def __calc_capex(self) -> gp.LinExpr:
        if self.__u is None:
            self.__u = self.__usage_vars()
        net_capex = gp.quicksum(self.__u["links"][(l.source, l.destination)]*l.capex for l in self.__links) + \
            gp.quicksum(self.__u["switches"][s.id] *
                        s.capex for s in self.__switches)
        control_capex = gp.quicksum(
            self.__x[s.id]*s.capex_cnt for s in self.__switches)
        deploy_capex = gp.quicksum(self.__u["hosts"]
                                   [h.id]*h.capex for h in self.__hosts)
        capex = net_capex+control_capex+deploy_capex
        return capex

    def __calc_opex(self) -> gp.LinExpr:
        if self.__u is None:
            self.__u = self.__usage_vars()
        net_opex = gp.quicksum(self.__u["links"][(l.source, l.destination)]*l.opex for l in self.__links) + gp.quicksum(
            self.__u["switches"][s.id]*s.opex for s in self.__switches)
        control_opex = gp.quicksum(
            self.__x[s.id]*s.opex_cnt for s in self.__switches)
        deploy_opex = gp.quicksum(self.__u["cycles"][h.id]*h.opex_cycle +
                                  self.__u["memory"][h.id]*h.opex_memory for h in self.__hosts)
        opex = net_opex+control_opex+deploy_opex
        return opex

    def __add_capex_budget(self, budget: float):
        if self.__capex is None:
            self.__capex = self.__calc_capex()

        self.__mip_model.addConstr(self.__capex <= budget, "capex_budget")

    def __add_opex_budget(self, budget: float):
        if self.__opex is None:
            self.__opex = self.__calc_opex()

        self.__mip_model.addConstr(self.__opex <= budget, "opex_budget")
    
    def __inner_max_rt_constraint(self, t: Workflow) -> None:
        if t.max_response_time > 0:
            wf_rt = 0
            for m in t.chain:
                for h in self.__hosts:  # Execution time
                    wf_rt += (self.__z[h.id][t.id][m] *
                              self.__dkt_microservices[m].cycles)/h.power
                for h in self.__hosts:
                    for l in self.__links:  # Traffic latency
                        wf_rt += self.__f[(l.source, l.destination)
                                          ][h.id][t.id][m]*l.latency
                        if l.destination in self.__dkt_switches:  # Control latency derived from traffic flows
                            wf_rt += gp.quicksum(self.__cf[l2.source, l2.destination]
                                                 [l.destination]*l2.latency for l2 in self.__links)
            if t.response:
                for h in self.__hosts:
                    for l in self.__links:  # Response traffic latency
                        wf_rt += self.__fprime[(l.source, l.destination)
                                               ][h.id][t.id]*l.latency
                        if l.destination in self.__dkt_switches:  # Control latency derived from response flows
                            wf_rt += gp.quicksum(self.__cf[l2.source, l2.destination]
                                                 [l.destination]*l2.latency for l2 in self.__links)
            self.__mip_model.addConstr(
                wf_rt <= t.max_response_time, 'qos_{}'.format(t.id))

    def __add_max_rt_constraint(self) -> "list[Future]":
        return [self.__executor.submit(self.__inner_max_rt_constraint, t=t) for t in self.__workflows]

    def objective_avg_response_time(self):
        self.__times['objective_start'] = perf_counter()
        self.__futures.append(self.__executor.submit(self.inner_objective_avg_response_time))

    def inner_objective_avg_response_time(self):
        self.__add_maximum_controllers_constraint()
        all_wf_rt = []
        for t in self.__workflows:
            wf_rt = 0
            for m in t.chain:
                for h in self.__hosts:  # Execution time
                    wf_rt += (self.__z[h.id][t.id][m] *
                                self.__dkt_microservices[m].cycles)/h.power
                for h in self.__hosts:
                    for l in self.__links:  # Traffic latency
                        wf_rt += self.__f[(l.source, l.destination)
                                          ][h.id][t.id][m]*l.latency
                        if l.destination in self.__dkt_switches:  # Control latency derived from traffic flows
                            wf_rt += gp.quicksum(self.__cf[l2.source, l2.destination]
                                                 [l.destination]*l2.latency for l2 in self.__links)
            if t.response:
                for h in self.__hosts:
                    for l in self.__links:  # Response traffic latency
                        wf_rt += self.__fprime[(l.source, l.destination)
                                               ][h.id][t.id]*l.latency
                        if l.destination in self.__dkt_switches:  # Control latency derived from response flows
                            wf_rt += gp.quicksum(self.__cf[l2.source, l2.destination]
                                                 [l.destination]*l2.latency for l2 in self.__links)
            all_wf_rt.append(wf_rt)

        #self.__mip_model.objective = minimize(gp.quicksum(all_wf_rt))
        self.__mip_model.setObjective(gp.quicksum(all_wf_rt), GRB.MINIMIZE)
        self.__times['objective_time'] = perf_counter()
    
    def objective_overall_cost(self):
        if self.__capex is None:
            self.__capex = self.__calc_capex()

        if self.__opex is None:
            self.__opex = self.__calc_opex()

        #self.__mip_model.objective = minimize(self.__capex+self.__opex)
        self.__mip_model.setObjective(self.__capex+self.__opex, GRB.MINIMIZE)

        self.__times['objective_time'] = perf_counter()

    def objective_capex(self):
        if self.__capex is None:
            self.__capex = self.__calc_capex()

        #self.__mip_model.objective = minimize(self.__capex)
        self.__mip_model.setObjective(self.__capex, GRB.MINIMIZE)

        self.__times['objective_time'] = perf_counter()

    def objective_opex(self):
        if self.__opex is None:
            self.__opex = self.__calc_opex()

        #self.__mip_model.objective = minimize(self.__opex)
        self.__mip_model.setObjective(self.__opex, GRB.MINIMIZE)

        self.__times['objective_time'] = perf_counter()

    def multiobjective(self):
        rtime = gp.quicksum(
            ((self.__z[h.id][t.id][m]*self.__dkt_microservices[m].cycles)/h.power) +
            (self.__f[(l.source, l.destination)][h.id][t.id][m]*l.latency) +
            (self.__fprime[(l.source, l.destination)][h.id][t.id]*l.latency) +
            (gp.quicksum(self.__cf[l2.source, l2.destination][l.destination] *
                         l2.latency for l2 in self.__links) if l.destination in self.__dkt_switches else 0)
            for h in self.__hosts for l in self.__links for t in self.__workflows for m in t.chain
        )

        if self.__capex is None:
            self.__capex = self.__calc_capex()

        if self.__opex is None:
            self.__opex = self.__calc_opex()

        #self.__mip_model.objective = minimize(self.__capex+self.__opex+rtime)
        self.__mip_model.setObjective(
            self.__capex+self.__opex+rtime, GRB.MINIMIZE)

        self.__times['objective_time'] = perf_counter()
    
    def save_debug(self, out_filename: str = 'TempDADO.lp') -> None:
        wait(self.__futures)
        self.__mip_model.write(out_filename)
    
    def execute(self):
        wait(self.__futures)
        self.__times['constraint_creation'] = perf_counter()
        self.__mip_model.update()
        #self.__mip_model.Params.NumericFocus = 3
        self.__mip_model.optimize()
        opt_status = self.__mip_model.getAttr(GRB.Attr.Status)
        if opt_status == GRB.INFEASIBLE:
            print('Infeasible problem!!!')
        else:
            self.__times['optimized'] = perf_counter()
            self.__exportable = True
            if opt_status != GRB.OPTIMAL:
                self.__suboptimal = True
        self.__optimized = True
    
    def preset_vars(self, var_vals: dict):
        for var in var_vals:
            m_var = self.__mip_model.var_by_name(var)
            self.__mip_model.add_constr(m_var == var_vals[var], 'accel_{}'.format(var))
    
    def get_solution(self) -> pd.DataFrame:
        if not self.__optimized:
            self.execute()
        if self.__exportable:
            sol = [{"Variable": var.varName, "Value": var.x}
                   for var in self.__mip_model.getVars()]
            return pd.DataFrame(sol)
        else:
            raise RuntimeError(
                'Trying to export solution of infeasible problem')
    
    def get_timing_report(self) -> dict:
        report = {}
        if 'loading' in self.__times:
            report['Scenario load'] = self.__times['loading'] - \
                self.__times['startup']
            if 'variable_creation' in self.__times:
                report['Variable creation'] = self.__times['variable_creation'] - \
                    self.__times['loading']
                if 'constraint_creation' in self.__times:
                    report['Constraint creation'] = self.__times['constraint_creation'] - \
                        self.__times['variable_creation']
                    if 'objective_start' in self.__times:
                        report['Objective creation'] = self.__times['objective_start'] - \
                            self.__times['constraint_creation']
        if 'optimized' in self.__times:
            if 'objective_time' in self.__times:
                report['Optimization'] = self.__times['optimized'] - \
                    self.__times['constraint_creation']
            report['Full process'] = self.__times['optimized'] - \
                self.__times['startup']
        if self.__suboptimal:
            report['Suboptimal solution!'] = 1.0
        return report

class CommandUI:

    AVG_RESPONSE_TIME_OBJ = 'avg_response_time'
    OVERALL_COST_OBJ = 'overall_cost'
    CAPEX_OBJ = 'capex'
    OPEX_OBJ = 'opex'
    MULTI_OBJ = 'multi'

    OBJECTIVES = [AVG_RESPONSE_TIME_OBJ, OVERALL_COST_OBJ, CAPEX_OBJ, OPEX_OBJ, MULTI_OBJ]

    def __init__(self):
        self.__parser = argparse.ArgumentParser(description='DADO MIP Optimizer')
        self.__init_parser()

    def __init_parser(self):
        self.__parser.add_argument('-i', required=True, metavar='JSON config', help='Case study or config in JSON format')
        self.__parser.add_argument('-obj', required=True, help='Optimization objective', choices=self.OBJECTIVES)
        self.__parser.add_argument('-o', required=True, metavar='CSV solution', help='File to output the solution to in CSV format')
        self.__parser.add_argument('-trep', required=False, metavar='CSV timing report', help='File to output the timing report to in CSV format')
        self.__parser.add_argument(
            '-debug', action='store_true', help='Enable debug')
    
    def launch(self):
        from file_io import ConfigLoader, TimingReport
        l_args = self.__parser.parse_args()
        loader = ConfigLoader(l_args.i)
        config = loader.load_json_config()
        dado_model = ParallelDADO(config)
        if l_args.obj == self.AVG_RESPONSE_TIME_OBJ:
            dado_model.objective_avg_response_time()
        elif l_args.obj == self.OVERALL_COST_OBJ:
            dado_model.objective_overall_cost()
        elif l_args.obj == self.CAPEX_OBJ:
            dado_model.objective_capex()
        elif l_args.obj == self.OPEX_OBJ:
            dado_model.objective_opex()
        elif l_args.obj == self.MULTI_OBJ:
            dado_model.multiobjective()
        else:
            raise RuntimeError('Invalid objective')
        if l_args.debug:
            dado_model.save_debug()
        dado_model.execute()
        dado_model.get_solution().to_csv(l_args.o, index=False)
        if l_args.trep:
            t_report = TimingReport(l_args.trep)
            t_report.export_as_csv(dado_model.get_timing_report())

if __name__ == "__main__":
    CommandUI().launch()
