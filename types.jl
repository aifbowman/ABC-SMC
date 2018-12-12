struct gsummary
  moms::Array{Float64}
  wts::Array{Float64}
  size::Int64
  lens::Array{Float64,1}
  threshold::Float64
end
 struct gsummary_scans
  moms::Array{Float64}
  wts::Array{Float64}
  size::Array{Int64}
  lens::Array{Float64,1}
  threshold::Float64
end
struct gsummary_cuts
  moms::Array{Float64}
  wts::Array{Float64}
  size::Int64
  cut::Vector{Vector{Vector{Float64}}}
  lens::Array{Float64,1}
  threshold::Float64
end
struct gsummary_sub
  moms::Array{Float64}
  wts::Array{Float64}
  size::Array{Int64,3}
  lens::Array{Float64,1}
  threshold::Float64
end

#model structure
struct Model
  parameters::Array{String,1}
  pdens::Function
  psampler::Function
end

# ABC algorithm output structure
struct ABCfit
  pts::Array{Any,2}
  sig::Array{Any,2}
  wts::Array{Any,2}
  p::Array{Float64,2}
  its::Array{Int64,1}
  dists::Array{Float64,2}
  epsilon::Array{Float64,1}
  temp::Array{Float64,2}
  pacc::Array{Float64}
  names::Array{Any,1}
  models::Array{Any,1}
end

# struct ABCfit
#   pts::Array{Any,2}
#   sig::Array{Any,2}
#   wts::Array{Any,2}
#   p::Array{Float64,2}
#   its::Array{Int64,1}
#   dists::Array{Float64,2}
#   epsilon::Array{Float64,1}
#   temp::Array{Float64,2}
#   pacc::Array{Float64}
# end
