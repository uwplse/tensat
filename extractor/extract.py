from __future__ import print_function
from ortools.linear_solver import pywraplp
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Construct and solve ILP')
    parser.add_argument('--time_lim_sec', type=int, default=10, metavar='N',
        help='Time limit in seconds (default: 10)')
    parser.add_argument('--order_var_int', action='store_true', default=False,
        help='Use integer variable for t')
    parser.add_argument('--class_constraint', action='store_true', default=False,
        help='Add constraint that each eclass sum to 1')

    return parser.parse_args()


def main():
    # Parse arguments
    args = get_args()

    # Load ILP data
    with open('./tmp/ilp_data.json') as f:
        data = json.load(f)

    costs = data['cost_i']
    e = data['e_m']
    h = data['h_i']
    g = data['g_i']
    root_m = data['root_m']
    num_nodes = len(costs)
    num_classes = len(e)

    epsilon = 1 / (10 * num_classes)
    if args.order_var_int:
        A = num_classes
    else:
        A = 2

    # Create solver
    solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    #solver.SetNumThreads(8)
    solver.SetTimeLimit(args.time_lim_sec * 1000)
    infinity = solver.infinity()

    # Define variables
    x = {}
    for j in range(num_nodes):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)

    t = {}
    if args.order_var_int:
        print("Use int var for order")
        for j in range(num_classes):
            t[j] = solver.IntVar(0, num_classes-1, 't[%i]' % j)
    else:
        for j in range(num_classes):
            t[j] = solver.NumVar(0.0, 1.0, 't[%i]' % j)

    print('Number of variables =', solver.NumVariables())

    # Define constraints
    # Root
    solver.Add(sum([x[j] for j in e[root_m]]) == 1)

    if args.class_constraint:
        print("Add class constraints")
        for m in range(num_classes):
            solver.Add(sum([x[j] for j in e[m]]) <= 1)
    
    for i in range(num_nodes):
        for m in h[i]:
            # Children
            solver.Add(sum([x[j] for j in e[m]]) - x[i] >= 0)
            # Order
            if args.order_var_int:
                solver.Add(t[g[i]] - t[m] + A * (1 - x[i]) >= 1)
            else:
                solver.Add(t[g[i]] - t[m] - epsilon + A * (1 - x[i]) >= 0)

    # Define objective
    obj_expr = [costs[j] * x[j] for j in range(num_nodes)]
    solver.Minimize(sum(obj_expr))

    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        #for j in range(num_nodes):
        #    print(x[j].name(), ' = ', x[j].solution_value())
        #for j in range(num_classes):
        #    print(t[j].name(), ' = ', t[j].solution_value())
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
    else:
        print('The problem does not have an optimal solution.')

    # Store results
    solved_x = [int(x[j].solution_value()) for j in range(num_nodes)]
    result_dict = {}
    result_dict["solved_x"] = solved_x
    result_dict["cost"] = solver.Objective().Value()
    with open('./tmp/solved.json', 'w') as f:
        json.dump(result_dict, f)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()