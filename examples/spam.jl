include("../RustGAM.jl")
using Main.RustGAM
# using PGFPlots

#load data...
data = readdlm("datasets/spam.data")


function guess_lambda()
    features, labels = data[:,1:end-1], vec(data[:, end])
    feature_names = [replace(strip(s),"_"," ") for s in readlines(open("datasets/spam.desc"))]
    labels[labels.==0.0] = -1.0
    test_ind = vec(map(Bool, readdlm("datasets/spam.traintest")))

    train_labels = labels[!test_ind]
    test_labels = labels[test_ind]
    train_features = features[!test_ind, :]
    #log-transform and normalize features.
    train_features = log.(train_features .+ 0.1)

    n_val = 100
    rp  = randperm(size(train_features,1))
    train_features = train_features[rp,:]

    ### split into train and cross val
    val_features = train_features[1:n_val,:]
    train_features = train_features[n_val+1:end,:]
    val_labels = train_labels[1:n_val]
    train_labels = train_labels[n_val+1:end]

    #standardize to [0,1]
    lb,ub = computebounds(train_features, 0.00,1.0)
    train_features = standardize(train_features, lb, ub)# (train_features.-mins)./(maxes-mins)
    val_features = standardize(val_features, lb, ub)


    n,d = size(train_features);

    @time path = log_space_reg_path(train_features,
    train_labels, 1.0, 10.0,1.02, LogisticLoss(); use_offset = true, n_tau = 250);

    err_path = [
    begin
      o = [gam(vec(val_features[i,:])) for i = 1:size(val_features,1)]
        labels_hat = sign(o)
        correct = sum(labels_hat.==val_labels)
        err_rate = (length(val_labels)-correct)/length(val_labels)
        println(tau, " ", length(gam.splines), " ", err_rate)
        (tau, err_rate)
    end for (tau, gam) in path[100:end]
    ]
    return err_path[indmin([e for (t,e) in err_path])][1]
end

cv_lambdas = [guess_lambda() for i in 1:1]
println("CV lambda:", mean(cv_lambdas))


features, labels = data[:,1:end-1], vec(data[:, end])
feature_names = [replace(strip(s),"_"," ") for s in readlines(open("datasets/spam.desc"))]
labels[labels.==0.0] = -1.0
test_ind = vec(map(Bool, readdlm("datasets/spam.traintest")))

train_labels = labels[!test_ind]
test_labels = labels[test_ind]
train_features = features[!test_ind, :]
#log-transform and normalize features.
train_features = log.(train_features .+ 0.1)

#standardize to [0,1]
lb,ub = computebounds(train_features, 0.00,1.0)
train_features = standardize(train_features, lb, ub)# (train_features.-mins)./(maxes-mins)

# test data
test_features = features[test_ind,:]
test_features = log.(features[test_ind,:] .+ 0.1)
test_features = standardize(test_features, lb, ub)

n,d = size(train_features);

@time path = log_space_reg_path(train_features,
train_labels, 1.0, 10.0,1.02, LogisticLoss(); use_offset = true, n_tau = 250);


err_path = [
begin
  o = [gam(vec(test_features[i,:])) for i = 1:size(test_features,1)]
    labels_hat = sign(o)
    correct = sum(labels_hat.==test_labels)
    println(tau, " ", length(gam.splines))
    err_rate = (length(test_labels)-correct)/length(test_labels)
    (tau, err_rate)
end for (tau, gam) in path[100:end]
]



  #   #   # plot coordiante functions
  #   #   group_plot_ph = GroupPlot(4, 4,  groupStyle = "horizontal sep = 1cm, vertical sep = 1.5cm")
  #   #   offset = apply_gam(gam,zeros(length(feature_names)))
  #   #
  #   #   responses = coordinate_responses(gam, length(feature_names), linspace(-0.1,1.1,100))
  #   #   r_max = [maxabs(r) for r in responses]
  #   #   for f_idx in reverse(sortperm(r_max)[end-15:end])
  #   #     response = responses[f_idx]
  #   #     println(response)
  #   #     color = all(response.==0.0) ? "blue" : "red"
  #   #     fit = Axis(Plots.Linear(linspace(-0.1,1.1,100), response, mark="none", style=color), title=feature_names[f_idx], ymin=-5.0, ymax = 5.0,width = "4cm")
  #   #     push!(group_plot_ph, fit)
  #   #   end
  #   #   save("spam/$i.pdf", group_plot_ph)
  #   #   save("spam/$i.tex", group_plot_ph)


#try glmnet
using GLMNet, Distributions
const y_glm = zeros(length(train_labels),2)
y_glm[find(train_labels.==1.0),1] = 1.0
y_glm[find(train_labels.!=1.0),2] = 1.0
glm_path = glmnet(train_features, y_glm, Binomial(); lambda_min_ratio=0.00001, nlambda=200)
predictions = predict(glm_path, test_features)
err_rates = Float64[sum((predictions[:,i].>= 0.0).==(test_labels.==1.0))/length(test_labels) for i = 1:size(predictions,2)]
glm_err_rate = minimum(err_rates)*100.0

#
# fit = Axis([
# Plots.Linear(Float64[p[1] for p in err_path], Float64[p[2] for p in err_path]*100.0, mark="none", legendentry="Smoothing Splines"),
# Plots.Linear(x -> glm_err_rate, (0,4000), legendentry="GLMNet", mark="none", style="dashed"),
# Plots.Linear(x -> 5.3, (0,4000), legendentry="Smoothing Splines", mark="none", style="dotted"),
# Plots.Linear(x -> 5.5, (0,4000), legendentry="GAMSEL", mark="none", style="dashdotted")
# ], title="Validation Error (\\%)", xlabel="\$\\tau\$", ylabel="Error Rate")
# save("spam/validation.pdf", fit)
# save("spam/validation.tex", fit)



include("../cvx_l1.jl")
using Convex, SCS

const n_disc = 20;
const knot_points = linspace(0.0,1.0,n_disc)'
const A = zeros(n,d*n_disc)
for idx = 1:n
  A[idx, :] = vec(eval_hinges(train_features[idx,:], knot_points))
end
const B = train_features
tau = 0.1;
test_mse = Inf;
A_test = zeros(size(test_features,1),d*n_disc)
for idx = 1:size(test_features,1)
  A_test[idx,:] = vec(eval_hinges(test_features[idx,:], knot_points))
end


#try glmnet with nonlinear coordinate functions
using GLMNet, Distributions
const y_glm = zeros(length(train_labels),2)
y_glm[find(train_labels.==1.0),1] = 1.0
y_glm[find(train_labels.!=1.0),2] = 1.0
glm_path = glmnet([A B], y_glm, Binomial(), penalty_factor=[ones(size(A,2));zeros(size(B,2))], standardize=false, intercept=false)
predictions = predict(glm_path, [A_test test_features])
err_rates = Float64[sum((predictions[:,i].>= 0.0).==(test_labels.==1.0))/length(test_labels) for i = 1:size(predictions,2)]
glm_err_rate = minimum(err_rates)*100.0
#
# # compile cvx?
# # solve_quad_l1(randn(10,10),randn(10),randn(10), 1.0);
# y = train_labels;
# x_v = Variable(size(A,2))
# z_v = Variable(size(B,2))
# o_v = Variable(1)
# obj = logisticloss(-y.*(A * x_v +B*z_v .+o_v))
# problem = minimize(obj, [sumabs(x_v) <= tau])
#
#
# results = Tuple{Float64,Float64}[]
# for i = 20:120
#   if i > 20
#     solve!(problem, warmstart=true, SCSSolver(verbose=true))
#   else
#     solve!(problem, SCSSolver(verbose=true))
#   end
#   (knot_w, lin_term, offset) = (x_v.value, z_v.value, o_v.value)
#
#   o = A_test*knot_w .+ test_features*lin_term .+ offset
#
#   labels_hat = sign(o)
#   correct = sum(labels_hat.==test_labels)
#
#   err_rate = (length(test_labels)-correct)/length(test_labels)
#   # if test_mse_new <= test_mse
#   #   x_input = linspace(-0.1,1.1,100)
#   #   o = max(x_input.-knot_points',0.0)*knot_w .+ x_input*lin_term[1] .+ offset
#   #   save("bone/fit_spline.pdf",Axis([Plots.Scatter(x, y, markSize=0.5), Plots.Linear(x_input, vec(o), mark="none")], title="Adaptive Spline"))
#   # end
#   println(tau, ": t_err: ", err_rate);
#   push!(results, (tau, err_rate))
#   tau *= 1.3
# end
