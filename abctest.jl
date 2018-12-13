using StatsBase
using Distributions
using Distributed
using Plots
using LsqFit
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
expd_lens=Array{Any}(undef,2)
expd_lens[1]=Array{Float64}(undef,2,5)
bootstrp=Array{Any}(undef,1000)
bootstrp[1]=d2[:]
for i in 2:length(bootstrp)
  bootstrp[i]=d2[sample(1:size(d2)[1],size(d2)[1],replace=true)]
end
temp=Array{Float64}(undef,5,1000)
for j in 1:1000
  temp[1,j]=mean(bootstrp[j])
  for i in 2:5
  temp[i,j]=mean((bootstrp[j].-mean(bootstrp[j])).^i)
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
  return(sqrt(mean([((mean(d)-expd[1][1,1])/expd[1][2,1])^2,((mean((d.-mean(d)).^2)-expd[1][1,2])/expd[1][2,2])^2,((mean((d.-mean(d)).^3)-expd[1][1,3])/expd[1][2,3])^2])))
end
rho_lens(expd_lens,[0,10,1,0.05,0.1])
#define model and priors for inference
model_lens=vcat(Uniform(0,2),Uniform(0,20),Uniform(0,20),Uniform(0,1),Uniform(0,0.2))
#do inference
# parameters: number of particles, data to be fit to, vector of models, vector of error functions, termination parameter (the smaller
# you choose the longer fitting runs, can set to zero), numbers of workers to spawn (set to 0 in case parellel computing not set up)
test_lens=APMC(100,expd_lens,Vector[model_lens],[rho_lens],paccmin=0.01)
#plot bivariate posterior of a vs b
scatter(test_lens.pts[1,end][1,:],test_lens.pts[1,end][2,:])

#now let's try a model selection example, with an automatically chosen number of particles:
#we'll use the adaptive initial number of particles APMC- APMC_KDE_adpt_init)
#we'll compete the full noisy linear map with an adder model (a fixed at 0):
model_lens_adder=vcat(Uniform(0,20),Uniform(0,20),Uniform(0,1),Uniform(0,0.2))
function rho_lens_adder(expd,d2)
  d=nlm(vcat(0,d2),expd[2],30,0.01,true)
  return(sqrt(mean([((mean(d)-expd[1][1,1])/expd[1][2,1])^2,((mean((d.-mean(d)).^2)-expd[1][1,2])/expd[1][2,2])^2,((mean((d.-mean(d)).^3)-expd[1][1,3])/expd[1][2,3])^2])))
end
test_lens_adpt=APMC_KDE_adpt_init(100,expd_lens,Vector[model_lens,model_lens_adder],[rho_lens,rho_lens_adder],ecv=0.2,paccmin=0.02)
# the console output after the fit termiantes should show that model 2 (adder) is favoured
# this makes sense as the parameters used to generate the synethetic data had a=0

#again, let's plot the bivariate posterior of a vs b from the full model
scatter(test_lens_adpt.pts[1,end][1,:],test_lens_adpt.pts[1,end][2,:])
# and the histogram of the posterior of b in the adder model
histogram(test_lens_adpt.pts[2,end][1,:])
