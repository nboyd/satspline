include("../RustGAM.jl")
using Main.RustGAM
# using PGFPlots

#load data...
data,header = readdlm("datasets/bone.data",header=true)
female = find(data[:,3].=="female")
data = data[female, :]
x = Float64[x for x in vec(data[:,2])]
y = Float64[x for x in vec(data[:,4])]
# hold out ~10% of data.
srand(18)
rp = randperm(length(x))
test_ind, train_ind  = rp[1:120], rp[121:end]
x_test = x[test_ind]
y_test = y[test_ind]
x = x[train_ind]
y = y[train_ind]
lb,ub = computebounds(x, 0.00,1.0)
x = standardize(x, lb, ub)
x_test = standardize(x_test, lb, ub)
n = length(x)


#scatter raw data
# raw_scatter = Axis(Plots.Scatter(x, y, markSize=0.5), xlabel="Standardized Age", ylabel="Relative Change in Bone Mineral Density")
# save("bone/raw_scatter.pdf", raw_scatter)
# save("bone/raw_scatter.tex", raw_scatter)
#

lambda = 0.005;
tol = 1E-2;

@time path = log_space_reg_path(reshape(x,length(x),1),y,
1E-2, 0.005, 1.03, SquaredLoss(); n_tau = 300);
#
# group_plot_tau = GroupPlot(1, 3,  groupStyle = "horizontal sep = 2cm, vertical sep = 2cm")
# rmse = Float64[];
# for (i,(tau, gam)) in enumerate(path)
#   o_test = [gam([f]) for f in x_test]
#   test_rmse = sqrt(mean((o_test .- y_test).^2))
#   rmse = [rmse; test_rmse]
#   println(i)
#   println("test rmse: ", test_rmse)
#   if i in [140, 220, 300]
#     (str,dstr,deltastr) = spline_str(gam.splines[1])
#     println(str)
#     println(dstr)
#     println(delta)
#     println(" offset: $(gam.offset)")
#     o = Float64[gam([f]) for f in linspace(-0.1,1.1,100)]
#     tau_string = @sprintf("%0.2f", tau)
#     fit = Axis([
#       Plots.Scatter(x, y, markSize=0.5),
#       Plots.Linear(vec(linspace(-0.1,1.1,100)), o, mark="none")],
#      title="\$\\tau\$ = $tau_string")
#     push!(group_plot_tau, fit)
#   end
# end
# save("bone/group_tau.pdf", group_plot_tau)
# save("bone/group_tau.tex", group_plot_tau)
# println("BEST ind: ", indmin(rmse), " ", minimum(rmse))


# test robustness to outliers with pseudo_huber
n_outliers = 30;
x_outliers = rand(n_outliers)
y_outliers = rand(n_outliers)
x = [x; x_outliers]
y = [y; y_outliers]

lambda = 0.005;
tol = 1E-2;

path = log_space_reg_path(reshape(x,length(x),1),y,
1E-2, 0.005, 1.03, SquaredLoss(); n_tau = 300);
# group_plot_robust = GroupPlot(2, 1, groupStyle = "horizontal sep = 2cm, vertical sep = 2cm")
# for (i, (tau, gam)) in enumerate(path)
#   if i==145
#     o = Float64[gam([f]) for f = linspace(-0.1,1.1,100)]
#     o_test = Float64[gam([f]) for f in x_test]
#     test_rmse = sqrt(mean((o_test .- y_test).^2))
#     tau_string = @sprintf("%0.2f", tau)
#     push!(group_plot_robust, Axis([Plots.Scatter(x, y, markSize=0.5), Plots.Linear(linspace(-0.1,1.1,100), o, mark="none")], title="Least Squares with \$\\tau\$ = $tau_string"))
#   end
# end
# exit(0);
#
#
# path = log_space_reg_path(reshape(x,length(x),1),y,
# 1E-2, 0.005, 1.03, HuberLoss(0.0015); n_tau = 300);
# for (i, (tau, gam)) in enumerate(path)
#   if i==215
#     o = Float64[gam([f]) for f = linspace(-0.1,1.1,100)]
#     o_test = Float64[gam([f]) for f in x_test]
#     test_rmse = sqrt(mean((o_test .- y_test).^2))
#     tau_string = @sprintf("%0.2f", tau)
#     push!(group_plot_robust, Axis([Plots.Scatter(x, y, markSize=0.5), Plots.Linear(linspace(-0.1,1.1,100), o, mark="none")], title="Pseudo-Huber with \$\\tau\$ = $tau_string"))
#   end
# end
# save("bone/group_robust.pdf", group_plot_robust)
# save("bone/group_robust.tex", group_plot_robust)
#
# #
### try fitting with standard adaptive splines
include("../cvx_l1.jl")
# const knot_points = x # linspace(0.0,1.0,100)
# const A = max(x .- knot_points',0.0)
# const B = reshape(deepcopy(x),length(x),1)
# tau = 0.005;
# test_rmse = Inf;
# x_var = Variable(size(A,2))
# z_var = Variable(size(B,2))
# o_var = Variable(1)
# problem = minimize(sumsquares((A * x_var +B*z_var) .+o_var - y), [sumabs(x_var) <= tau])
# for i = 20:500
#   if i > 1
#     solve!(problem, warmstart=true)
#   else
#     solve!(problem)
#   end
#
#   (knot_w, lin_term, offset) = (x_var.value, z_var.value, o_var.value))
#   o_test = max(x_test .-knot_points',0.0)*knot_w .+ x_test*lin_term[1] .+ offset
#   test_rmse_new = sqrt(mean((o_test .- y_test).^2))
#   if i >= 100 && i % 20 == 0
#     x_input = linspace(-0.1,1.1,100)
#     o = max(x_input.-knot_points',0.0)*knot_w .+ x_input*lin_term[1] .+ offset
#     # save("bone/fit_spline_$i.pdf",Axis([Plots.Scatter(x, y, markSize=0.5), Plots.Linear(x_input, vec(o), mark="none")], title="Adaptive Spline"))
#   end
#   test_rmse = test_rmse_new
#   println(tau, ": t_rmse: ", test_rmse);
#   tau *= 1.03
# end
