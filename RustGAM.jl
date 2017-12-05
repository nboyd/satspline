include("common.jl")

module RustGAM
  import Base.call
  export
    LossFn, SquaredLoss, LogisticLoss, HuberLoss,
    approximate_regularization_path,log_space_reg_path,
    spline_str, fit_gam
  export GAM, Spline, coordinate_responses,coordinate_functions
  export call
  const libname = Base.Sys.isapple() ? "liblasso.dylib" : "liblasso.so"
  const libpath = abspath(joinpath(dirname(@__FILE__),  "target","release", libname))
  print("Loading $libname... ")
  const library = Libdl.dlopen(libpath)
  const mk_ts = Libdl.dlsym(library, :make_training_dataset)
  const mk_g = Libdl.dlsym(library, :make_gam)

  const f_g = Libdl.dlsym(library, :fit_gam)
  const apply_g = Libdl.dlsym(library, :apply_gam)
  const num_k = Libdl.dlsym(library, :num_knots)
  const get_k = Libdl.dlsym(library, :get_knots)
  # const compute_score = Libdl.dlsym(library, :compute_score)
  println("loaded.")

  abstract type LossFn end
  struct SquaredLoss <: LossFn end
  struct LogisticLoss <: LossFn end
  struct HuberLoss <: LossFn
    delta :: Float64
  end

  loss_type_to_rust_enum(l :: SquaredLoss) = ccall(Libdl.dlsym(library, :return_sq_loss),Ptr{Void},())
  loss_type_to_rust_enum(l :: LogisticLoss) = ccall(Libdl.dlsym(library, :return_log_loss),Ptr{Void},())
  loss_type_to_rust_enum(l :: HuberLoss) = ccall(Libdl.dlsym(library, :return_huber_loss),Ptr{Void}, (Float64,), l.delta)


  function make_training_set(A :: Matrix{Float64})
    (n,d) = size(A);
    A_vecs = Vector{Float64}[vec(A[i,:]) for i = 1:n]
    td = ccall(mk_ts,Ptr{Void},
    (Ptr{Ptr{Float64}}, UInt64, UInt64),
    A_vecs,n, d)
  end

  function make_gam(d :: Int64)
    gam = ccall(mk_g,Ptr{Void},
    (UInt64,),
    d)
  end

  function fit_gam!(gam, training_data,y,  lambda :: Float64, tol :: Float64, loss :: LossFn; use_offset :: Bool = true)
    r_loss = loss_type_to_rust_enum(loss)
    ccall(f_g,Float64,
    (Ptr{Void}, Ptr{Void}, Ptr{Float64}, Float64, Float64, Bool, Ptr{Void}),
    gam, training_data, y, lambda, tol, use_offset, r_loss);
  end

    function fit_gam(A, y, tau :: Float64, tol :: Float64, loss :: LossFn; use_offset :: Bool = true)
        @assert all(0.0 .<= A .<= 1.0)
        td = make_training_set(A);
        gam = make_gam(size(A,2));
        path = Tuple{Float64, GAM}[];
        score = fit_gam!(gam, td, y,  tau, tol, loss; use_offset = use_offset)
        # translate to julia gam and store in path;
        return gam_rusttojulia(gam)
    end

  function get_knots(gam)
    k = ccall(num_k,UInt64,(Ptr{Void},),gam);
    f_idx = zeros(UInt64,k)
    locations = zeros(Float64,k)
    weights = zeros(Float64,k)
    ccall(get_k,Void, (Ptr{Void},Ptr{UInt64},Ptr{Float64},Ptr{Float64}), gam, f_idx, locations, weights)
    (map(Int64,f_idx).+1, locations, weights)
  end

get_offset(gam) =
    ccall(Libdl.dlsym(library, :offset),Float64,(Ptr{Void}, ),gam)

  function apply_gam(gam, x :: Vector{Float64})
    ccall(apply_g,Float64,(Ptr{Void}, Ptr{Float64}, UInt64),gam,x,length(x))
  end

  # function computescore(training_data,y, loss :: LossFn; use_offset :: Bool = true)
  #   ccall(compute_score,Ptr{Void},
  #   (Ptr{Void}, Ptr{Float64},Ptr{Void}, Float64, Bool),
  #   training_data,y,loss_type_to_rust_enum(loss), use_offset);
  # end

  function gam_rusttojulia(gam)
      (f_idx, locations, weights) = get_knots(gam)
      offset = get_offset(gam)
      nonzero_idx = unique(f_idx)
      GAM([Spline(idx, locations[find(f_idx .== idx)], weights[find(f_idx .== idx)]) for idx in nonzero_idx], offset)
  end

  function approximate_regularization_path(A :: Matrix{Float64}, y, tol :: Float64, m :: Float64,
    loss :: LossFn; use_offset :: Bool = true, n_tau :: Int64 = 100, max_knots :: Int64 = 100)
    @assert tol > 0.0
    @assert m > 1.0
    @assert max_knots > 0
    @assert n_tau > 0
    @assert all(0.0 .<= A .<= 1.0)

    td = make_training_set(A);
    gam = make_gam(size(A,2));
    tau = 0.0
    path = Tuple{Float64, GAM}[];
    for iter = 1:n_tau
      score = fit_gam!(gam, td, y,  tau, tol/m, loss; use_offset = use_offset)
      # translate to julia gam and store in path;
      j_gam = gam_rusttojulia(gam)
      push!(path, (tau, j_gam))
      println("iter\t$iter\ttau\t$tau\tknots\t", numknots(j_gam));
      tau = tau + (1.0 - 1.0/m)*tol/abs(score);
    end

    path
  end

  function log_space_reg_path(A :: Matrix{Float64}, y, tol :: Float64, tau_init :: Float64, tau_mul :: Float64,
    loss :: LossFn; use_offset :: Bool = true, n_tau :: Int64 = 100, max_knots :: Int64 = 100)
    @assert tol > 0.0
    @assert tau_init > 0.0
    @assert tau_mul > 1.0
    @assert n_tau > 0
    @assert all(0.0 .<= A .<= 1.0)

    td = make_training_set(A);
    gam = make_gam(size(A,2));
    tau = tau_init
    path = Tuple{Float64, GAM}[];
    for iter = 1:n_tau
      score = fit_gam!(gam, td, y,  tau, tol, loss; use_offset = use_offset)
      # translate to julia gam and store in path;
      j_gam = gam_rusttojulia(gam)
      push!(path, (tau, j_gam))
      println("iter\t$iter\ttau\t$tau\tknots\t", numknots(j_gam));
      tau = tau*tau_mul;
    end

    path
  end

  # function cross_validate(A :: Matrix{Float64}, y, tol :: Float64, tau_init :: Float64, tau_mul :: Float64,
  #   loss :: LossFn; use_offset :: Bool = true, n_tau :: Int64 = 100, max_knots :: Int64 = 100)

  struct Spline
      f_idx :: Int64
      locations :: Vector{Float64}
      weights :: Vector{Float64}
  end

  struct GAM
    splines :: Vector{Spline}
    offset :: Float64
  end

  function coordinate_responses(gam, num_f :: Int64, v)
    responses = Array(Array{Float64},num_f)
    for f_idx = 1:num_f
      input = zeros(num_f)
      response = zeros(length(v))
      s_idx = findfirst(s -> s.f_idx == f_idx, gam.splines)
      if s_idx != 0.0
        s = gam.splines[s_idx]
        for (idx,x) in enumerate(v)
          response[idx] = s(x)
        end
      end
      responses[f_idx] = response
    end
    responses
  end


  function spline_str(s :: Spline)
    str = ""
    deriv_str = "(-0.1,0) "
    delta_str = ""
    w_sum = 0.0
    for (w,l) in zip(s.weights,s.locations)
      str = string(str, "$w*max(x-$l,0) + ")
      deriv_str  = string(deriv_str," ($l,$w_sum) ($l,$(w_sum+w)) ")
      w_sum += w
      delta_str = string(delta_str, " ($l, $w) ")
    end
    (str, string(deriv_str," (1.1,0.0)"), delta_str)
  end

  numknots(g :: GAM) = sum(Int64[length(s.weights) for s in g.splines])

  (s :: Spline)(v :: Float64) =
    dot(s.weights, max.(v .- s.locations,0.0)) :: Float64

  (g :: GAM)(v :: Vector{Float64}) =
    sum(Float64[s(v[s.f_idx]) for s in g.splines]) + g.offset :: Float64


    # (s :: Spline)(v :: Vector{Float64}) =
    #   dot(s.weights, max.(v .- s.locations,0.0)) :: Float64
    #
    # (g :: GAM)(d :: Matrix{Float64}) = #### (n,d)
    #   sum((s(d[:,s.f_idx]) for s in g.splines)) + g.offset :: Float64

end
