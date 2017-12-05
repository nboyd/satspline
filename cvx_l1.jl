using Convex, SCS
####
#
# min_x,z ||Ax + Bz + 1*o - y||^2
# s.t.
# |x|_1 \le tau

function solve_quad_l1(A,B,y,tau)
  n = size(A,2);
  m = size(B,2);
  d = size(A,1);
  x = Variable(n)
  z = Variable(m)
  o = Variable(1)


  problem = minimize(sumsquares((A * x +B*z) .+o - y), [sumabs(x) <= tau])

  # Solve the problem by calling solve!
  solve!(problem,SCSSolver(verbose=false))
  # Check the status of the problem
  problem.status # :Optimal, :Infeasible, :Unbounded etc.
  # Get the optimal value
  (x.value,z.value, o.value)
end


function eval_hinges(x, t)
  if size(t,2) == 1
    t = t'
  end
  max(x .- t, 0.0)
end
