using StatsBase
using Distributions
using PyPlot
import PyPlot.plot
import Distributions.rand
import Distributions.pdf
include("nlm.jl")
include("types.jl")
include("functions.jl")
include("abc2.jl")


#create synthetic data using noisy linear map
#arguments: vector of noisy linear map paramters (a,b,additive noise, division noise, growth rate variability), number of cells, number of cell cycles, time step in cell cyles
d2=nlm(vcat(0,10,1,0.05,0.1),1000,30,0.01)
#store moments from sample and generate weights
expd_lens=Array(Any,2)
expd_lens[1]=Array(Float64,2,5)
bootstrp=Array(Any,1000)
bootstrp[1]=d2[:]
for i in 2:length(bootstrp)
  bootstrp[i]=d2[sample(1:size(d2)[1],size(d2)[1],replace=true)]
end
temp=Array(Float64,5,1000)
for j in 1:1000
  temp[1,j]=mean(bootstrp[j])
  for i in 2:5
  temp[i,j]=mean((bootstrp[j]-mean(bootstrp[j])).^i)
  end
end
for i in 1:5
  expd_lens[1][1,i]=temp[i,1]
  expd_lens[1][2,i]=sqrt(var(temp[i,:]))
end
expd_lens[2]=length(d2)
#error function for length distribution using first three moments
function rho_lens(expd,d2)
  d=nlm(d2,expd[2],30,0.01,true)
  return(sqrt(mean([((mean(d)-expd[1][1,1])/expd[1][2,1])^2,((mean((d-mean(d)).^2)-expd[1][1,2])/expd[1][2,2])^2,((mean((d-mean(d)).^3)-expd[1][1,3])/expd[1][2,3])^2])))
end
rho_lens(expd_lens,[0.14500452,10.99259240,0.82982153,0.02742845,0.1])
#define model and priors for inference
model_lens=vcat(Uniform(0,2),Uniform(0,20),Uniform(0,20),Uniform(0,1),Uniform(0,0.2))
#do inference
# parameters: number of particles, data to be fit to, vector of models, fitting parameter (keep at 0.5), vector of error functions, termination parameter (the smaller
# you choose the longer fitting runs, can set to zero), fitting parameter (keep at 2)
test_lens=APMC(100,expd_lens,Vector[model_lens],0.5,[rho_lens],0.01,2)
#plot the histograms of univariate posterior estimates
plot(test_lens)
#plot bivariate posterior of a vs b
scatter(test_lens.pts[1,end][1,:],test_lens.pts[1,end][2,:])
