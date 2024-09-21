from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver("GLOP")

x = solver.NumVar(0, solver.infinity(), 'x')
y = solver.NumVar(0, solver.infinity(), 'y')
z = solver.NumVar(0, solver.infinity(), 'z')

print(solver.NumVariables())

solver.Add(50*x + 6*y + 3*z >= 200)
solver.Add(30*x + 5*y + 5*z >= 300)


solver.Minimize(2000*x+300*y+200*z)

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
	print("Minimum:", solver.Objective().Value())
	x = x.solution_value()
	y = y.solution_value()
	z = z.solution_value()
	print("x:", x)
	print("y:", y)
	print("z:", z)

print((50*x + 6*y + 3*z))
print(30*x + 5*y + 5*z)
	
solver = pywraplp.Solver.CreateSolver("GLOP")

x = solver.NumVar(0, solver.infinity(), 'x')
y = solver.NumVar(0, solver.infinity(), 'y')

print(solver.NumVariables())

solver.Add(50*x + 30*y <= 2000)
solver.Add(6*x + 5*y <= 275.1)
solver.Add(3*x + 5*y <= 200)


solver.Maximize(200*x+300*y)

status = solver.Solve()


if status == pywraplp.Solver.OPTIMAL:
	print("Maximize:", solver.Objective().Value())
	print("x:", x.solution_value())
	print("y:", y.solution_value())


print(50*5/8 + 3*56.25)
print(30*5/8 + 5*56.25)

225/4
