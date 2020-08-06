from __future__ import print_function
from ortools.linear_solver import pywraplp
import json


def main():
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
    A = 2

    # Create solver
    solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    infinity = solver.infinity()

    # Define variables
    x = {}
    for j in range(num_nodes):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)

    t = {}
    for j in range(num_classes):
        t[j] = solver.NumVar(0.0, 1.0, 't[%i]' % j)

    print('Number of variables =', solver.NumVariables())

    # Define constraints
    # Root
    solver.Add(sum([x[j] for j in e[root_m]]) == 1)
    
    for i in range(num_nodes):
        for m in h[i]:
            # Children
            solver.Add(sum([x[j] for j in e[m]]) - x[i] >= 0)
            # Order
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


'''
def create_data_model():
  """Stores the data for the problem."""
  data = {}
  data['constraint_coeffs'] = [
      [5, 7, 9, 2, 1],
      [18, 4, -9, 10, 12],
      [4, 7, 3, 8, 5],
      [5, 13, 16, 3, -7],
  ]
  data['bounds'] = [250, 285, 211, 315]
  data['obj_coeffs'] = [7, 8, 2, 9, 6]
  data['num_vars'] = 5
  data['num_constraints'] = 4
  return data


def main():
  data = create_data_model()
  # Create the mip solver with the CBC backend.

  for i in range(data['num_constraints']):
    constraint = solver.RowConstraint(0, data['bounds'][i], '')
    for j in range(data['num_vars']):
      constraint.SetCoefficient(x[j], data['constraint_coeffs'][i][j])
  print('Number of constraints =', solver.NumConstraints())
  # In Python, you can also set the constraints as follows.
  # for i in range(data['num_constraints']):
  #  constraint_expr = \
  # [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
  #  solver.Add(sum(constraint_expr) <= data['bounds'][i])

  objective = solver.Objective()
  for j in range(data['num_vars']):
    objective.SetCoefficient(x[j], data['obj_coeffs'][j])
  objective.SetMaximization()
  # In Python, you can also set the objective as follows.
  # obj_expr = [data['obj_coeffs'][j] * x[j] for j in range(data['num_vars'])]
  # solver.Maximize(solver.Sum(obj_expr))

  status = solver.Solve()

  if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    for j in range(data['num_vars']):
      print(x[j].name(), ' = ', x[j].solution_value())
    print()
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
  else:
    print('The problem does not have an optimal solution.')


if __name__ == '__main__':
  main()
'''