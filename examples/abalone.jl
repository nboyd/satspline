include("../RustGAM.jl")
using Main.RustGAM
# using PGFPlots
data = readdlm("datasets/abalone")
feature_names = [strip(s) for s in readlines(open("datasets/abalone.desc"))]
y_mins = Float64[-2, -3, -1, -1,-1,-18,-5,-1]
y_maxes = Float64[1,1,5,3,25,1,1,12]
y_mins[:] = -15.0;
y_maxes[:] = 15;

age = map(Float64,data[:,1])
features = data[:,2:end]
data = map(d -> parse(Float64,split(d,":")[2]),features)

tm,tstd = mean(data,1), std(data,1)
train_features = (data.-tm)./tstd
srand(100)
rp = randperm(size(train_features,1))
test_ind, train_ind  = rp[1:400], rp[401:end]

test_features = train_features[test_ind,:]
test_y = age[test_ind]
train_features = train_features[train_ind, :]
age = age[train_ind]

lb,ub = computebounds(train_features, 0.00, 1.00)
# maxes, mins = minimum(train_features,1), maximum(train_features,1)
#standardize to [0,1]
train_features = standardize(train_features, lb, ub)
test_features = standardize(test_features, lb, ub)

lambda = 1.0;
tol = 1.0;



# ### try py-earth
# using PyCall
# @pyimport pyearth as Earth
# py_e = Earth.Earth();
# py_e[:fit](train_features, age)
# println(py_e[:summary]())

const (n,d) = size(train_features)
# path = log_space_reg_path(train_features,
# age, 10.0, 20.0, 10.0, SquaredLoss(); use_offset = true, n_tau = 3)

path = log_space_reg_path(train_features,
age, 10.0, 2.0, 1.1, SquaredLoss(); use_offset = true, n_tau = 72)


for (tau, gam) in path
  o = [gam(vec(test_features[i,:])) for i = 1:size(test_features,1)]
  test_loss = sqrt(mean((o - test_y).^2))
  println("tau: ", tau,  " loss: ", test_loss)
end
#
# for (i,(tau, gam)) in enumerate(path)
#   group_plot_ph = GroupPlot(4, 2,  groupStyle = "horizontal sep = 1.5cm, vertical sep = 1.5cm")
#   responses = coordinate_responses(gam, 8, linspace(-0.1,1.1,100))
#   for (f_idx, response) in enumerate(responses)
#     color = all(response.==0.0) ? "blue" : "red"
#     fit = Axis(Plots.Linear(linspace(-0.1,1.1,100), response, mark="none", style=color), title=feature_names[f_idx], ymin=y_mins[f_idx], ymax = y_maxes[f_idx],width = "3cm")
#     push!(group_plot_ph, fit)
#   end
#   pushPGFPlotsPreamble("\\pgfplotsset{width=10cm,compat=1.9}")
#   fname = @sprintf "abalone/%03d.pdf" i
#   save(fname, group_plot_ph)
#
#   # save("abalone/$i.tex", group_plot_ph)
# end
