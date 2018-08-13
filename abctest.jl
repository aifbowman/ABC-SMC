rmprocs(workers())

save("ws.jld", "rpb12_complete_nogl_o3", rpb12_complete_nogl_o3)
@everywhere function rho_lens2(expd,d2)
  d=nlm(d2,expd[2],30,0.01,true)
  if length(d)!=length(unique(d))
    return(Inf)
  else
  return(sqrt(mean([((mean(d)-expd[1][1,1])/expd[1][2,1])^2,((mean((d-mean(d)).^2)-expd[1][1,2])/expd[1][2,2])^2,((mean((d-mean(d)).^3)-expd[1][1,3])/expd[1][2,3])^2,((mean((d-mean(d)).^4)-expd[1][1,4])/expd[1][2,4])^2,((mean((d-mean(d)).^5)-expd[1][1,5])/expd[1][2,5])^2])))
end
end
@everywhere function rho_lens_log(expd,d2)
  d=nlm(d2,2132,30,0.01,true)
  if length(d)!=length(unique(d))
    return(Inf)
  else
  return(mean([log(((mean(d)-expd[1,1]))^2),log(((mean((d-mean(d)).^2)-expd[1,2]))^2),log(((mean((d-mean(d)).^3)-expd[1,3]))^2),log(((mean((d-mean(d)).^4)-expd[1,4]))^2),log(((mean((d-mean(d)).^5)-expd[1,5]))^2)]))
end
end
d2=rpb1_raw
expd_lens=Array(Any,2)
expd_lens[1]=Array(Float64,2,5)
bootstrp=Array(Any,1000)
bootstrp[1]=d2[:,2]
for(i in 2:length(bootstrp))
  bootstrp[i]=d2[sample(1:size(d2)[1],size(d2)[1],replace=true),2]
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
expd_lens[2]=length(d2[:,2])
rho_lens2(expd_lens,[0.14500452,10.99259240,0.82982153,0.02742845,0.1])
test_lens=APMC(1000,d2[:,2],[model_lens],0.5,[rho_lens],0.01,2)
test_lens2=APMC(1000,expd_lens,[model_lens],0.5,[rho_lens2],0.01,2)
test_lens3=APMC(1000,expd_lens,[model_lens],0.5,[rho_lens3],0.01,2)
test_lens4=APMC(1000,expd_lens,[model_lens],0.5,[rho_lens4],0.01,2)
test_lens_log=APMC(1000,expd_lens,[model_lens],0.5,[rho_lens_log],0.01,2)

for i in 1:2, j in 1:2
  subplot(2,2,2*(i-1)+j)
  plot(kde(collect(test_lens.pts[1,end][end-4+(2*(i-1)+j),:])),label="1D ks distance (76 iterations)")
  plot(kde(collect(test_lens4.pts[1,end][end-4+(2*(i-1)+j),:])),label="1D first 2 moments (71 iterations)")
  plot(kde(collect(test_lens3.pts[1,end][end-4+(2*(i-1)+j),:])),label="1D first 3 moments (93 iterations)")
  plot(kde(collect(test_lens2.pts[1,end][end-4+(2*(i-1)+j),:])),label="1D first 5 moments (77 iterations)")
  plot(kde(collect(test_lens_log.pts[1,end][end-4+(2*(i-1)+j),:])),label="1D log first 5 moments (92 iterations)")
  plot(kde(collect(test_pois2.pts[1,end][end-4+(2*(i-1)+j),:])),label="2D moments of order =<5 (108 iterations)")
  plot([params[2*(i-1)+j+1],params[2*(i-1)+j+1]],[0,maximum(kde(collect(test_lens.pts[1,end][end-4+(2*(i-1)+j),:])).density)])
  PyPlot.xlabel(model_lens.parameters[end-4+(2*(i-1)+j)])
  legend(loc="upper right",fancybox="true")
end
suptitle("posteriors of noisy linear map parameters fitted to synthetic data")

scatter(collect(test_lens5.pts[1,end][1,:]),collect(test_lens5.pts[1,end][2,:]))
scatter(collect(test_lens.pts[1,end][1,:]),collect(test_lens.pts[1,end][2,:]),color="red",alpha=0.1)
using JLD
save("synthlenfits.jld", "2x10^3", test, "2X10^4", test2, "2x10^5", test3)
scatter(test.pts[1,75][1,:],test.pts[1,75][3,:],color="red",alpha=0.5,color="blue",label="2x10^3 cells")
scatter(test2.pts[1,75][1,:],test2.pts[1,75][3,:],color="blue",alpha=0.5,color="green",label="2x10^4 cells")
scatter(test3.pts[1,84][1,:],test3.pts[1,84][3,:],color="red",alpha=0.5,label="2x10^5 cells")
PyPlot.xlabel("a (memory)")
PyPlot.ylabel("sd of noise")
legend(loc="upper right",fancybox="true")
nms=8
binomial(nms+1+ 3 -1,3)

dropna(constit[:,1].*constit[:,5])
N=100
s=50
np=vcat(1,1)
temp=transpose(hcat(vcat(fill(1,50),fill(2,50)),1:100,1:100))
inds=sortperm(reshape(temp[end,:],N))[1:s]
temp=temp[:,inds]
pts=Array{Any}(2)
for j in 1:2
  pts[j]=temp[2:(np[j]+1),find(temp[1,:].==j)]
  if size(pts[j])[2]>0
    keep=inds[find(reshape(temp[1,:].==j,s))].<=s
    wts[j,i]= collect(@parallel vcat for k in 1:length(keep)
      if !keep[k]
        pdf(models[j],(pts[j,i][:,k]))/pdf(ker[j,i-1],pts[j,i][:,k])
      else
        0.0
      end
    end)
    l=1
    for k in 1:length(keep)
      if keep[k]
        wts[j,i][k]=wts[j,i-1][l]
        l=l+1
      end
    end
    wts[j,i]=weights(wts[j,i])
  else
    wts[j,i]=zeros(0)
  end
end

temp=APMC_KDE_adpt(N,expd,models,rho,ecv=0.1)


temp=APMC_KDE_adpt_init(N,expd_NB,models,rho,ecv=0.1)
temp=APMC_KDE_adpt_init(N,expd_P,models[1:1],rho[1:1],ecv=0.1)
temp=APMC_KDE_adpt_init(N,expd_NB,models[2:2],rho[2:2],ecv=0.1)


N=5000
expd_NB= rand(NegativeBinomial(30,0.3),1000)
expd_P= rand(Poisson(60),1000)


expd=expd_P
models=Vector[vcat(Uniform(0,100)),vcat(Uniform(0,100),Uniform(0,1))]
rho=[(x,y)->ApproximateTwoSampleKSTest(x+rand(Normal(0,0.001),length(x)),rand(Poisson(y[1]),1000)+rand(Normal(0,0.001),length(x))).δ,(x,y)->ApproximateTwoSampleKSTest(x+rand(Normal(0,0.001),length(x)),rand(NegativeBinomial(y[1],y[2]),1000)+rand(Normal(0,0.001),length(x))).δ]
names=Vector[[string("parameter",i) for i in 1:length(models[m])] for m in 1:length(models)]
prop=0.5
paccmin=0.02
ecv=0.1
B=10
wks=4-length(workers())

@time ecv_solver(N1,ker[:,i],pts[:,i],wts[:,i],np,lm,B,ecv)
@time ecv_solver_opt(N1,ker[:,i],pts[:,i],wts[:,i],np,lm,B,ecv)
32270038
MixtureMode
