include("../RustGAM.jl")
using Main.RustGAM
#using PGFPlots

#
# ### try py-earth
# using PyCall
# @pyimport pyearth as Earth
# py_e = Earth.Earth();
# py_e[:fit](train_features, train_y)



function cv_tau()
    n_val = 100
    train_features, h = readdlm("datasets/als/trainX.csv", ',';header=true)
    rp  = randperm(size(train_features,1))
    train_features = train_features[rp,:]
    val_features = train_features[1:n_val,:]
    train_features = train_features[n_val+1:end,:]
    train_y, h = readdlm("datasets/als/trainY.csv", ',';header=true)
    train_y = train_y[rp,:]
    val_y = train_y[1:n_val,2]
    train_y = train_y[n_val+1:end,2]


    test_features, h = readdlm("datasets/als/testX.csv", ',';header=true)
    #first column is ID...
    train_features = train_features[:,2:end]
    val_features = val_features[:,2:end]
    test_features = test_features[:,2:end]

    test_y, h = readdlm("datasets/als/testY.csv", ',';header=true)
    #train_y = vec(train_y[:,2])
    test_y = vec(test_y[:,2])
    println(size(train_features))

    #tm,tstd = mean(train_features,1), std(train_features,1)
    #train_features = (train_features.-tm)./tstd
    lb,ub = computebounds(train_features, 0.00, 1.00)
    #standardize to [0,1]
    train_features = standardize(train_features, lb, ub)
    #standardize test data
    #test_features = (test_features.-tm)./tstd
    test_features = standardize(test_features, lb, ub)
    #val_features = (val_features.-tm)./tstd
    val_features = standardize(val_features, lb, ub)

    const n = size(train_features,1)
    const d = size(train_features,2)

    @time path = approximate_regularization_path(train_features, train_y, 1.0, 100.0,
      SquaredLoss(); use_offset = true, n_tau = 130)

    err_path =[
    begin
      o = [gam(vec(val_features[i,:])) for i = 1:size(val_features,1)]
      test_loss = mean((o - val_y).^2)
      println("tau: ", tau,  " loss: ", test_loss)
      (tau, test_loss)
    end
     for (tau, gam) in path[70:end]
    ]

    return err_path[indmin([y for (x,y) in err_path])][1]
end

cv_taus = [cv_tau() for i in 1:1]
tau_star = mean(cv_taus)


train_features, h = readdlm("datasets/als/trainX.csv", ',';header=true)
train_y, h = readdlm("datasets/als/trainY.csv", ',';header=true)
train_y = train_y[:,2]
test_features, h = readdlm("datasets/als/testX.csv", ',';header=true)
#first column is ID...
train_features = train_features[:,2:end]
test_features = test_features[:,2:end]

test_y, h = readdlm("datasets/als/testY.csv", ',';header=true)
test_y = vec(test_y[:,2])
println(size(train_features))


lb,ub = computebounds(train_features, 0.00, 1.00)
#standardize to [0,1]
train_features = standardize(train_features, lb, ub)
#standardize test data
test_features = standardize(test_features, lb, ub)

const n = size(train_features,1)
const d = size(train_features,2)

gam = fit_gam(train_features, train_y, tau_star, 1.0/100.0, SquaredLoss())

o = [gam(vec(test_features[i,:])) for i = 1:size(test_features,1)]
test_loss = mean((o - test_y).^2)

using DecisionTree
model = build_forest(train_y,train_features, 5, 20, 5, 1.0)
predictions = apply_forest(model, test_features)
rmse = sqrt(mean((predictions.-test_y).^2))

include("../cvx_l1.jl")
const n_disc = 20;
const knot_points = linspace(0.0,1.0,n_disc)'
const A = zeros(n,d*n_disc)
for idx = 1:n
  A[idx, :] = vec(eval_hinges(train_features[idx,:], knot_points))
end
const A_test = zeros(size(test_features,1),d*n_disc)
for idx = 1:size(test_features,1)
  A_test[idx,:] = vec(eval_hinges(test_features[idx,:], knot_points))
end
const B = train_features
# B = zeros(size(B,1),0)

using GLMNet
glm_path = glmnet([A B], train_y, penalty_factor=[ones(size(A,2));zeros(size(B,2))], standardize=false, intercept=false, nlambda=100,dfmax=1000, lambda=[0.32,0.3,0.25,0.2,0.15,0.1,0.05,0.01,0.005,0.003,0.0025,0.001,0.0005])
predictions = GLMNet.predict(glm_path, [A_test test_features])
mse = mean((predictions.-test_y).^2, 1)

using GLMNet
glm_path = glmnet(train_features, train_y; lambda_min_ratio=0.00001, nlambda=200)
predictions = GLMNet.predict(glm_path, test_features)
mse = mean((predictions.-test_y).^2, 1)
#
# fit = Axis(
# [Plots.Linear(Float64[p[1] for p in err_path], Float64[p[2] for p in err_path], mark="none", legendentry="Saturating Splines")
# Plots.Linear(x -> minimum(mse), (1,17), legendentry="GLMNet", mark="none", style="dashed")], title="ALS Regression", xlabel="\$\\tau\$", ylabel="Validation MSE")
# save("als/validation.pdf", fit)
# save("als/als_validation.tex", fit)
