
function computebounds(X, min_percentile, max_percentile)
  assert(0 <= min_percentile < max_percentile <= 1.0)
  sorted = sort(X,1)
  n = size(X,1)
  min_idx = round(Int,ceil(min_percentile*n + 1E-5))
  max_idx = round(Int,floor(max_percentile*n))
  sorted[min_idx,:], sorted[max_idx,:]
end

function standardize(X, min_v, max_v)
  X_new = deepcopy(X)
  for i = 1:size(X,2)
    minv = min_v[i]
    maxv = max_v[i]
    if minv == maxv
        X_new[:, i] = 0.0
    else
        c = -minv
        m = c + maxv
        X_new[:, i] = (X_new[:, i].+c)./m
    end
  end
  min.(max.(X_new,0.0),1.0)
end
