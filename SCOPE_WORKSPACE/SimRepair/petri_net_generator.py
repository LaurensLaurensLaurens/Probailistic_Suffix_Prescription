from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net import properties
from pm4py.objects.petri_net.utils import petri_utils as utils
from pm4py.objects.petri_net.obj import Marking
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.conversion.log import converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer

#CREATE PETRI NET
nodes_to_ignore = []
decision_nodes = []
decorations = {}
def create_PN(net):
    source = PetriNet.Place("source")
    sink = PetriNet.Place("sink")
    net.places.add(source)
    net.places.add(sink)

    # Init + G_Procedure
    t_init = PetriNet.Transition("initiate_case", "initiate_case")
    t_prior = PetriNet.Transition("start_priority", "start_priority")
    t_stan = PetriNet.Transition("start_standard", "start_standard")

    # Priority branch
    t_wait_emp = PetriNet.Transition("wait_for_employee", "wait_for_employee")
    t_ghost_take_emp = PetriNet.Transition("ghost_take_employee", "ghost_take_employee")
    t_ghost_reject_emp = PetriNet.Transition("ghost_reject_employee", "ghost_reject_employee")
    t_spare = PetriNet.Transition("use_spare_parts", "use_spare_parts")
    t_prior_rep = PetriNet.Transition("repair_priority", "repair_priority")
    t_prior_rework = PetriNet.Transition("prepare_rerepair_priority", "prepare_rerepair_priority")
    t_ghost_prior_good = PetriNet.Transition("ghost_prior_good", "ghost_prior_good")

    # Standard branch
    t_emp = PetriNet.Transition("choose_employee", "choose_employee")
    t_mat = PetriNet.Transition("choose_material", "choose_material")
    t_ghost_mat_forgotten = PetriNet.Transition("ghost_material_forgotten", "ghost_material_forgotten")
    t_ghost_mat_ordered = PetriNet.Transition("ghost_material_ordered", "ghost_material_ordered")
    t_stan_rep = PetriNet.Transition("repair_standard", "repair_standard")
    t_stan_rework = PetriNet.Transition("prepare_rerepair_standard", "prepare_rerepair_standard")
    t_ghost_stan_good = PetriNet.Transition("ghost_stan_good", "ghost_stan_good")

    # QC loop, shipping and end events
    t_qc = PetriNet.Transition("conduct_qc", "conduct_qc")
    t_improve = PetriNet.Transition("improve_rework", "improve_rework")
    t_ship = PetriNet.Transition("shipping", "shipping")
    t_acc = PetriNet.Transition("customer_acceptance", "customer_acceptance")
    t_ref = PetriNet.Transition("customer_refusal", "customer_refusal")

    t_list = [t_init, t_prior, t_stan,
              t_wait_emp, t_ghost_take_emp, t_ghost_reject_emp, t_spare, t_prior_rep, t_prior_rework, t_ghost_prior_good,
              t_emp, t_mat, t_ghost_mat_forgotten, t_ghost_mat_ordered, t_stan_rep, t_stan_rework, t_ghost_stan_good,
              t_qc, t_improve, t_ship, t_acc, t_ref]

    p_proc = PetriNet.Place("p_proc")                       # G_Procedure
    p_prior_wait = PetriNet.Place("p_prior_wait")           # merge of setup, re-repair and employee-wait loop
    p_emp_accept = PetriNet.Place("p_emp_accept")           # G_PriorityEmployeeAccept: take available employee?
    p_spare = PetriNet.Place("p_spare")
    p_prior_rep = PetriNet.Place("p_prior_rep")
    p_x1 = PetriNet.Place("p_x1")                           # G_X1: bad repair?
    p_stan_emp = PetriNet.Place("p_stan_emp")
    p_mat = PetriNet.Place("p_mat")                         # merge before material ordering
    p_x2 = PetriNet.Place("p_x2")                           # G_X2: material order forgotten?
    p_stan_rep = PetriNet.Place("p_stan_rep")
    p_x3 = PetriNet.Place("p_x3")                           # G_X3: bad repair?
    p_qc_decision = PetriNet.Place("p_qc_decision")         # G_Merge + agent decision: conduct (another) QC or ship?
    p_improve = PetriNet.Place("p_improve")
    p_accept = PetriNet.Place("p_accept")                   # G_Accept: customer accepts?
    p_list = [p_proc, p_prior_wait, p_emp_accept, p_spare, p_prior_rep, p_x1,
              p_stan_emp, p_mat, p_x2, p_stan_rep, p_x3,
              p_qc_decision, p_improve, p_accept]

    for p in p_list:
            net.places.add(p)

    for t in t_list:
        net.transitions.add(t)

    utils.add_arc_from_to(source, t_init, net)
    utils.add_arc_from_to(t_init, p_proc, net)
    utils.add_arc_from_to(p_proc, t_prior, net)
    utils.add_arc_from_to(p_proc, t_stan, net)

    # Priority branch
    utils.add_arc_from_to(t_prior, p_prior_wait, net)
    utils.add_arc_from_to(p_prior_wait, t_wait_emp, net)
    utils.add_arc_from_to(t_wait_emp, p_emp_accept, net)
    utils.add_arc_from_to(p_emp_accept, t_ghost_take_emp, net)
    utils.add_arc_from_to(t_ghost_take_emp, p_spare, net)
    utils.add_arc_from_to(p_emp_accept, t_ghost_reject_emp, net)
    utils.add_arc_from_to(t_ghost_reject_emp, p_prior_wait, net)
    utils.add_arc_from_to(p_spare, t_spare, net)
    utils.add_arc_from_to(t_spare, p_prior_rep, net)
    utils.add_arc_from_to(p_prior_rep, t_prior_rep, net)
    utils.add_arc_from_to(t_prior_rep, p_x1, net)
    utils.add_arc_from_to(p_x1, t_ghost_prior_good, net)
    utils.add_arc_from_to(t_ghost_prior_good, p_qc_decision, net)
    utils.add_arc_from_to(p_x1, t_prior_rework, net)
    utils.add_arc_from_to(t_prior_rework, p_prior_wait, net)

    # Standard branch
    utils.add_arc_from_to(t_stan, p_stan_emp, net)
    utils.add_arc_from_to(p_stan_emp, t_emp, net)
    utils.add_arc_from_to(t_emp, p_mat, net)
    utils.add_arc_from_to(p_mat, t_mat, net)
    utils.add_arc_from_to(t_mat, p_x2, net)
    utils.add_arc_from_to(p_x2, t_ghost_mat_forgotten, net)
    utils.add_arc_from_to(t_ghost_mat_forgotten, p_mat, net)
    utils.add_arc_from_to(p_x2, t_ghost_mat_ordered, net)
    utils.add_arc_from_to(t_ghost_mat_ordered, p_stan_rep, net)
    utils.add_arc_from_to(p_stan_rep, t_stan_rep, net)
    utils.add_arc_from_to(t_stan_rep, p_x3, net)
    utils.add_arc_from_to(p_x3, t_ghost_stan_good, net)
    utils.add_arc_from_to(t_ghost_stan_good, p_qc_decision, net)
    utils.add_arc_from_to(p_x3, t_stan_rework, net)
    utils.add_arc_from_to(t_stan_rework, p_stan_emp, net)

    # QC loop
    utils.add_arc_from_to(p_qc_decision, t_qc, net)
    utils.add_arc_from_to(t_qc, p_improve, net)
    utils.add_arc_from_to(p_improve, t_improve, net)
    utils.add_arc_from_to(t_improve, p_qc_decision, net)

    # Shipping and end events
    utils.add_arc_from_to(p_qc_decision, t_ship, net)
    utils.add_arc_from_to(t_ship, p_accept, net)
    utils.add_arc_from_to(p_accept, t_acc, net)
    utils.add_arc_from_to(t_acc, sink, net)
    utils.add_arc_from_to(p_accept, t_ref, net)
    utils.add_arc_from_to(t_ref, sink, net)

    net.initial_marking = Marking()
    net.initial_marking[source] = 1
    net.final_marking = Marking()
    net.final_marking[sink] = 1

    nodes_to_ignore = [t_ghost_take_emp, t_ghost_reject_emp, t_ghost_prior_good, t_ghost_mat_forgotten, t_ghost_mat_ordered, t_ghost_stan_good]
    for node in nodes_to_ignore:
        decorations[node] = {"color": "#E5E5E5"}

# VISUALIZE
def vizualize_net(net, format="png"):
        parameters = {"format": format, "decorations": decorations}
        gviz = pn_visualizer.apply(net, net.initial_marking, net.final_marking, parameters=parameters)
        pn_visualizer.view(gviz)
        pn_visualizer.save(gviz, "petri_net_unc.png")

#RUN
def generate_petri_net():
    petri_net =  PetriNet("Petri_net")
    create_PN(petri_net)
    # vizualize_net(petri_net)
    return petri_net
