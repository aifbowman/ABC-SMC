using Distributions
using LinearAlgebra
#rejection samplerm (for first iteration)
function init(models,expd,np,rho)
  d=Inf
  count=1
  m=sample(1:length(models))
  params=rand(models[m])
  d=rho[m](expd,params)
  # end
  return vcat(m,params,fill(0,maximum(np)-np[m]),d,count)
end

#SMC sampler (for subsequent iterations)
function cont(models,pts,wts,expd,np,i,ker,rho)
  d=Inf
  count=1
  #while d==Inf
  m=sample(1:length(models))
  while size(pts[m,i-1])[2]==0
    m=sample(1:length(models))
  end
  params=pts[m,i-1][:,sample(1:size(pts[m,i-1])[2],wts[m,i-1])]
  params=params+rand(ker[m])
  while pdf(models[m],params)==0
    m=sample(1:length(models))
    while size(pts[m,i-1])[2]==0
      m=sample(1:length(models))
    end
    count=count+1
    params=pts[m,i-1][:,sample(1:size(pts[m,i-1])[2],wts[m,i-1])]
    params=params+rand(ker[m])
  end
  d=rho[m](expd,params)
  # end
  return vcat(m,params,fill(0,maximum(np)-np[m]),d,count)
end

function APMC(N,expd,models,rho,;names=Vector[[string("parameter",i) for i in 1:length(models[m])] for m in 1:length(models)],prop=0.5,paccmin=0.02,n=2)
  i=1
  lm=length(models)
  s=round(Int,N*prop)
  #array for number of parameters in each model
  np=Array{Int64}(undef,length(models))
  for j in 1:lm
    np[j]=length(models[j])
  end
  #array for SMC kernel used in weights
  ker=Array{Any}(undef,lm)
  template=Array{Any}(undef,lm,1)
  #particles array
  pts=similar(template)
  #covariance matrix array
  sig=similar(template)
  #weights array
  wts=similar(template)
  #model probability at each iteration array
  p=zeros(lm,1)
  temp=@distributed hcat for j in 1:N
    init(models,expd,np,rho)
  end
  its=[sum(temp[size(temp)[1],:])]
  epsilon=[quantile(collect(temp[maximum(np)+2,:]),prop)]
  pacc=ones(lm,1)
  println(round.([epsilon[i];its[i]],digits=3))
  temp=temp[:,temp[maximum(np)+2,:].<=epsilon[i]]
  temp=temp[:,1:s]
  for j in 1:lm
    pts[j,i]=temp[2:(np[j]+1),temp[1,:].==j]
    wts[j,i]=StatsBase.weights(fill(1.0,sum(temp[1,:].==j)))
  end
  dists=transpose(temp[(maximum(np)+2),:])
  for j in 1:lm
    p[j]=sum(wts[j,1])
  end
  for j in 1:lm
    sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
  end
  p=p./sum(p)
  nbs=Array{Integer}(undef,length(models))
  for j in 1:lm
    nbs[j]=length(wts[j,i])
    println(round.(hcat(mean(diag(sig[j,i])[1:(np[j])]),pacc[j,i],nbs[j],p[j,i]),digits=3))
  end
  while maximum(pacc[:,i])>paccmin
    pts=reshape(pts,i*length(models))
    sig=reshape(sig,i*length(models))
    wts=reshape(wts,i*length(models))
    for j in 1:length(models)
      push!(pts,Array{Any}(undef,1))
      push!(sig,Array{Any}(undef,1))
      push!(wts,Array{Any}(undef,1))
    end
    pts=reshape(pts,length(models),i+1)
    sig=reshape(sig,length(models),i+1)
    wts=reshape(wts,length(models),i+1)
    i=i+1
    for j in 1:lm
      ker[j]=MvNormal(fill(0.0,np[j]),n*sig[j,i-1])
    end
    temp2=@distributed hcat for j in (1:(N-s))
      cont(models,pts,wts,expd,np,i,ker,rho)
    end
    its=vcat(its,sum(temp2[size(temp2)[1],:]))
    temp=hcat(temp,temp2)
    inds=sortperm(reshape(temp[maximum(np)+2,:],N))[1:s]
    temp=temp[:,inds]
    dists=hcat(dists,transpose(temp[(maximum(np)+2),:]))
    epsilon=vcat(epsilon,temp[(maximum(np)+2),s])
    pacc=hcat(pacc,zeros(lm))
    for j in 1:lm
        if sum(temp2[1,:].==j)>0
      pacc[j,i]=sum(temp[1,inds.>s].==j)/sum(temp2[1,:].==j)
      else pacc[j,i]==0
      end
    end
    println(round.(vcat(epsilon[i],its[i]),digits=3))
    for j in 1:lm
      pts[j,i]=temp[2:(np[j]+1),temp[1,:].==j]
      if size(pts[j,i])[2]>0
        keep=inds[reshape(temp[1,:].==j,s)].<=s
        wts[j,i]= @distributed vcat for k in 1:length(keep)
          if !keep[k]
            pdf(models[j],(pts[j,i][:,k]))/(1/(sum(wts[j,i-1]))*dot(values(wts[j,i-1]),pdf(ker[j],broadcast(-,pts[j,i-1],pts[j,i][:,k]))))
          else
            0.0
          end
        end
        if length(wts[j,i])==1
          wts[j,i]=fill(wts[j,i],1)
        end
        l=1
        for k in 1:length(keep)
          if keep[k]
            wts[j,i][k]=wts[j,i-1][l]
            l=l+1
          end
        end
          if length(wts[j,i])>1
        wts[j,i]=StatsBase.weights(wts[j,i])
          end
      else
        wts[j,i]=zeros(0)
      end
    end
    p=hcat(p,zeros(length(models)))
    for j in 1:lm
      p[j,i]=sum(wts[j,i])
    end
    for j in 1:lm
      if(size(pts[j,i])[2]>np[j])
        #sig[j,i]=cov(transpose(pts[j,i]),wts[j,i])
        sig[j,i]=cov(transpose(pts[j,i]))
        if isposdef(sig[j,i])
          dker=MvNormal(pts[j,i-1][:,1],n*sig[j,i])
          if pdf(dker,pts[j,i][:,1])==Inf
            sig[j,i]=sig[j,i-1]
          end
        else
          sig[j,i]=sig[j,i-1]
        end
      else
        sig[j,i]=sig[j,i-1]
      end
    end
    p[:,i]=p[:,i]./sum(p[:,i])
    for j in 1:lm
      nbs[j]=length(wts[j,i])
      println(round.(hcat(mean(diag(sig[j,i])./diag(sig[j,1])),pacc[j,i],nbs[j],p[j,i]),digits=3))
    end
  end
  samp=ABCfit(pts,sig,wts,p,its,dists,epsilon,temp,pacc,names,models)
  return(samp)
end


#SMC sampler (for subsequent iterations)
function cont_KDE(models,pts,wts,expd,np,i,ker,rho)
  d=Inf
  count=1
  #while d==Inf
  m=sample(1:length(models))
  while size(pts[m,i-1])[2]==0
    m=sample(1:length(models))
  end
  params=rand(ker[m,i-1])
  while pdf(models[m],params)==0
    m=sample(1:length(models))
    while size(pts[m,i-1])[2]==0
      m=sample(1:length(models))
    end
    count=count+1
  params=rand(ker[m,i-1])
  end
  d=rho[m](expd,params)
  # end
  return vcat(m,params,fill(0,maximum(np)-np[m]),d,count)
end


function APMC_KDE(N,expd,models,rho,;names=Vector[[string("parameter",i) for i in 1:length(models[m])] for m in 1:length(models)],prop=0.5,paccmin=0.02)
  i=1
  lm=length(models)
  s=round(Int,N*prop)
  #array for number of parameters in each model
  np=Array{Int64}(undef,length(models))
  for j in 1:lm
    np[j]=length(models[j])
  end
  template=Array{Any}(undef,lm,1)
  #particles array
  pts=similar(template)
  #covariance matrix array
  sig=similar(template)
  #weights array
  wts=similar(template)
  #model probability at each iteration array
  p=zeros(lm,1)
  #array for KDE estimates
  ker=similar(template)
  temp=@distributed hcat for j in 1:N
    init(models,expd,np,rho)
  end
  its=[sum(temp[size(temp)[1],:])]
  epsilon=[quantile(collect(temp[maximum(np)+2,:]),prop)]
  pacc=ones(lm,1)
  println(round.([epsilon[i];its[i]],digits=3))
  temp=temp[:,findall(temp[maximum(np)+2,:].<=epsilon[i])]
  temp=temp[:,1:s]
  for j in 1:lm
    pts[j,i]=temp[2:(np[j]+1),findall(temp[1,:].==j)]
    wts[j,i]=StatsBase.weights(fill(1.0,sum(temp[1,:].==j)))
  end
  dists=transpose(temp[(maximum(np)+2),:])
  for j in 1:lm
    p[j]=sum(wts[j,1])
  end
  for j in 1:lm
    sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
  end
  p=p./sum(p)
  nbs=Array{Integer}(undef,length(models))
  for j in 1:lm
    nbs[j]=length(wts[j,i])
    println(round.(hcat(mean(diag(sig[j,i])[1:(np[j])]),pacc[j,i],nbs[j],p[j,i]),digits=3))
  end
  for j in 1:lm
      mus=pts[j,i]
      neff=1/sum((wts[j,i].values./sum(wts[j,i])).^2)
      bs=(4/(neff*(np[j]+2)))^(1/(np[j]+4))
    ker[j,i]=MixtureModel(map(u->MvNormal(u,bs^2*sig[j,i]),[mus[:,i] for i in 1:size(mus)[2]]),wts[j,i].values/sum(wts[j,i]))
  end
  while maximum(pacc[:,i])>paccmin
    pts=reshape(pts,i*length(models))
    sig=reshape(sig,i*length(models))
    wts=reshape(wts,i*length(models))
    ker=reshape(ker,i*length(models))
    for j in 1:length(models)
      push!(pts,Array{Any}(undef,1))
      push!(sig,Array{Any}(undef,1))
      push!(wts,Array{Any}(undef,1))
      push!(ker,Array{Any}(undef,1))
    end
    pts=reshape(pts,length(models),i+1)
    sig=reshape(sig,length(models),i+1)
    wts=reshape(wts,length(models),i+1)
    ker=reshape(ker,length(models),i+1)
    i=i+1
    temp2=@distributed hcat for j in (1:(N-s))
      cont_KDE(models,pts,wts,expd,np,i,ker,rho )
    end
    its=vcat(its,sum(temp2[size(temp2)[1],:]))
    temp=hcat(temp,temp2)
    inds=sortperm(reshape(temp[maximum(np)+2,:],N))[1:s]
    temp=temp[:,inds]
    dists=hcat(dists,transpose(temp[(maximum(np)+2),:]))
    epsilon=vcat(epsilon,temp[(maximum(np)+2),s])
    pacc=hcat(pacc,zeros(lm))
    for j in 1:lm
        if sum(temp2[1,:].==j)>0
      pacc[j,i]=sum(temp[1,inds.>s].==j)/sum(temp2[1,:].==j)
      else pacc[j,i]==0
      end
    end
    println(round.(vcat(epsilon[i],its[i]),digits=3))
    for j in 1:lm
      pts[j,i]=temp[2:(np[j]+1),findall(temp[1,:].==j)]
      if size(pts[j,i])[2]>0
        keep=inds[reshape(temp[1,:].==j,s)].<=s
        wts[j,i]=@distributed vcat for k in 1:length(keep)
          if !keep[k]
            pdf(models[j],(pts[j,i][:,k]))/pdf(ker[j,i-1],pts[j,i][:,k])
          else
            0.0
          end
        end
        if length(wts[j,i])==1
          wts[j,i]=fill(wts[j,i],1)
        end
        l=1
        for k in 1:length(keep)
          if keep[k]
            wts[j,i][k]=wts[j,i-1][l]
            l=l+1
          end
        end
        if length(wts[j,i])>1
        wts[j,i]=StatsBase.weights(wts[j,i])
      end
      else
        wts[j,i]=zeros(0)
      end
    end
    p=hcat(p,zeros(length(models)))
    for j in 1:lm
      p[j,i]=sum(wts[j,i])
    end
    for j in 1:lm
      if(size(pts[j,i])[2]>np[j])
          mus=pts[j,i]
          neff=1/sum((wts[j,i].values./sum(wts[j,i])).^2)
          bs=(4/(neff*(np[j]+2)))^(1/(np[j]+4))
        sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
        #sig[j,i]=cov(transpose(pts[j,i]))
        if isposdef(bs^2*sig[j,i])
          dker=MvNormal(pts[j,i-1][:,1],bs^2*sig[j,i])
          if pdf(dker,pts[j,i][:,1])==Inf
            sig[j,i]=sig[j,i-1]
            ker[j,i]= ker[j,i-1]
          else
              #println(j,"OK")
              ker[j,i]=MixtureModel(map(u->MvNormal(u,bs^2*sig[j,i]),[mus[:,k] for k in 1:size(mus)[2]]),wts[j,i].values/sum(wts[j,i]))
          end
        else
          sig[j,i]=sig[j,i-1]
          ker[j,i]= ker[j,i-1]
        end
      else
        sig[j,i]=sig[j,i-1]
        ker[j,i]= ker[j,i-1]
      end
    end
    p[:,i]=p[:,i]./sum(p[:,i])
    for j in 1:lm
      nbs[j]=length(wts[j,i])
      println(round.(hcat(mean(diag(sig[j,i])./diag(sig[j,1])),pacc[j,i],nbs[j],p[j,i]),digits=3))
    end
  end
  samp=ABCfit(pts,sig,wts,p,its,dists,epsilon,temp,pacc,names,models)
  return(samp)
end

function APMC_KDE_adpt(N,expd,models,rho,;names=Vector[[string("parameter",i) for i in 1:length(models[m])] for m in 1:length(models)],prop=0.5,paccmin=0.02,ecv=0.1,B=10)
  i=1
  lm=length(models)
  N1=round(Int,N*prop)
  #array for number of parameters in each model
  np=Array{Int64}(undef,length(models))
  for j in 1:lm
    np[j]=length(models[j])
  end
  template=Array{Any}(undef,lm,1)
  #particles array
  pts=similar(template)
  #covariance matrix array
  sig=similar(template)
  #weights array
  wts=similar(template)
  #model probability at each iteration array
  p=zeros(lm,1)
  #array for KDE estimates
  ker=similar(template)
  temp=@distributed hcat for j in 1:N
    init(models,expd,np,rho)
  end
  its=[sum(temp[size(temp)[1],:])]
  epsilon=[quantile(collect(temp[maximum(np)+2,:]),prop)]
  pacc=ones(lm,1)
  println(round.([epsilon[i];its[i]],digits=3))
  temp=temp[:,findall(temp[maximum(np)+2,:].<=epsilon[i])]
  temp=temp[:,1:N1]
  for j in 1:lm
    pts[j,i]=temp[2:(np[j]+1),findall(temp[1,:].==j)]
    wts[j,i]=StatsBase.weights(fill(1.0,sum(temp[1,:].==j)))
  end
  dists=transpose(temp[(maximum(np)+2),:])
  for j in 1:lm
    p[j]=sum(wts[j,1])
  end
  for j in 1:lm
    sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
  end
  p=p./sum(p)
  nbs=Array{Integer}(undef,length(models))
  for j in 1:lm
    nbs[j]=length(wts[j,i])
    println(round.(hcat(mean(diag(sig[j,i])[1:(np[j])]),pacc[j,i],nbs[j],p[j,i]),digits=3))
  end
  for j in 1:lm
    mus=pts[j,i]
    neff=1/sum((wts[j,i].values./sum(wts[j,i])).^2)
    bs=(4/(neff*(np[j]+2)))^(1/(np[j]+4))
    ker[j,i]=MixtureModel(map(u->MvNormal(u,bs^2*sig[j,i]),[mus[:,i] for i in 1:size(mus)[2]]),wts[j,i].values/sum(wts[j,i]))
  end
  sizes=floor.(collect(range(N1/3,stop=N1*2,length=10)))
  ecvs=Array{Float64}(length(sizes))
  for j in 1:length(sizes)
    neff=sizes[j]
    bs=(4/(neff*(np[1]+2)))^(1/(np[1]+4))
    dens= @distributed vcat for b in 1:B
      boot=transpose(rand(ker[1,i],Int(sizes[j])))
      cv=cov(boot)
      MixtureModel(map(u->MvNormal(u,bs^2*cv),[boot[k,:] for k in 1:Int(sizes[j])]))
    end
    cvs= @distributed vcat for k in 1:size(pts[1,i])[2]
      vals= map(u->pdf(u,pts[1,i][:,k]),dens)
      std(vals)/mean(vals)
    end
    ecvs[j]=dot(cvs,wts[1,i].values)/sum(wts[1,i])
  end
  model(x,p)=p[1]*x.^(-p[2])
  fit=curve_fit(model,sizes,ecvs,[0.5,0.5])
  N2=Int(round((ecv/fit.param[1])^(1/(-fit.param[2]))))
  while maximum(pacc[:,i])>paccmin
    pts=reshape(pts,i*length(models))
    sig=reshape(sig,i*length(models))
    wts=reshape(wts,i*length(models))
    ker=reshape(ker,i*length(models))
    for j in 1:length(models)
      push!(pts,Array{Any}(undef,1))
      push!(sig,Array{Any}(undef,1))
      push!(wts,Array{Any}(undef,1))
      push!(ker,Array{Any}(undef,1))
    end
    pts=reshape(pts,length(models),i+1)
    sig=reshape(sig,length(models),i+1)
    wts=reshape(wts,length(models),i+1)
    ker=reshape(ker,length(models),i+1)
    i=i+1
    if (ceil(N2/prop)-N1)>0
      temp2=@distributed hcat for j in 1:Int(ceil(N2/prop)-N1)
        cont_KDE(models,pts,wts,expd,np,i,ker,rho)
      end
      its=vcat(its,sum(temp2[size(temp2)[1],:]))
      temp=hcat(temp,temp2)
    else
      its=vcat(its,0)
      temp=temp
    end
    inds=sortperm(reshape(temp[maximum(np)+2,:],size(temp)[2]))[1:N2]
    temp=temp[:,inds]
    dists=hcat(dists,transpose(temp[(maximum(np)+2),:]))
    epsilon=vcat(epsilon,temp[(maximum(np)+2),N2])
    pacc=hcat(pacc,zeros(lm))
    for j in 1:lm
      if (ceil(N2/prop)-N1)>0
        pacc[j,i]=sum(temp[1,inds.>N1].==j)/sum(temp2[1,:].==j)
      else
        pacc[j,i]=1
      end

    end
    println(round.(vcat(epsilon[i],its[i]),digits=3))
    for j in 1:lm
      pts[j,i]=temp[2:(np[j]+1),findall(temp[1,:].==j)]
      if size(pts[j,i])[2]>0
        keep=inds[findall(reshape(temp[1,:].==j,N2))].<=N1
        wts[j,i]=@distributed vcat for k in 1:length(keep)
          if !keep[k]
            pdf(models[j],(pts[j,i][:,k]))/pdf(ker[j,i-1],pts[j,i][:,k])
          else
            0.0
          end
        end
        l=1
        for k in 1:length(keep)
          if keep[k]
            wts[j,i][k]=wts[j,i-1][l]
            l=l+1
          end
        end
        wts[j,i]=StatsBase.weights(wts[j,i])
      else
        wts[j,i]=zeros(0)
      end
    end
    p=hcat(p,zeros(length(models)))
    for j in 1:lm
      p[j,i]=sum(wts[j,i])
    end
    for j in 1:lm
      if(size(pts[j,i])[2]>np[j])
        sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
        #sig[j,i]=cov(transpose(pts[j,i]))
        if isposdef(sig[j,i])
          dker=MvNormal(pts[j,i-1][:,1],sig[j,i])
          if pdf(dker,pts[j,i][:,1])==Inf
            sig[j,i]=sig[j,i-1]
          end
        else
          sig[j,i]=sig[j,i-1]
        end
      else
        sig[j,i]=sig[j,i-1]
      end
    end
    p[:,i]=p[:,i]./sum(p[:,i])
    for j in 1:lm
      if(size(pts[j,i])[2]>0)
        mus=pts[j,i]
        neff=1/sum((wts[j,i].values./sum(wts[j,i])).^2)
        bs=(4/(neff*(np[j]+2)))^(1/(np[j]+4))
        ker[j,i]=MixtureModel(map(u->MvNormal(u,bs^2*sig[j,i]),[mus[:,i] for i in 1:size(mus)[2]]),wts[j,i].values/sum(wts[j,i]))
      else
        ker[j,i]= ker[j,i-1]
      end
    end
    for j in 1:lm
      nbs[j]=length(wts[j,i])
      println(round.(hcat(mean(diag(sig[j,i])./diag(sig[j,1])),pacc[j,i],nbs[j],p[j,i]),digits=3))
    end
    sizes=floor.(collect(range(N2/3,N2*2,10)))
    ecvs=Array{Float64}(length(sizes))
    for j in 1:length(sizes)
      neff=sizes[j]
      bs=(4/(neff*(np[1]+2)))^(1/(np[1]+4))
      dens= @distributed vcat for b in 1:B
        boot=transpose(rand(ker[1,i],Int(sizes[j])))
        cv=cov(boot)
        MixtureModel(map(u->MvNormal(u,bs^2*cv),[boot[k,:] for k in 1:Int(sizes[j])]))
      end
      cvs= @distributed vcat for k in 1:size(pts[1,i])[2]
        vals= map(u->pdf(u,pts[1,i][:,k]),dens)
        std(vals)/mean(vals)
      end
      ecvs[j]=dot(cvs,wts[1,i].values)/sum(wts[1,i])
    end
    model(x,p)=p[1]*x.^(-p[2])
    fit=curve_fit(model,sizes,ecvs,[0.5,0.5])
    N1=N2
    N2=Int(round((ecv/fit.param[1])^(1/(-fit.param[2]))))
  end
  samp=ABCfit(pts,sig,wts,p,its,dists,epsilon,temp,pacc,names,models)
  return(samp)
end

function boot_dens(bs,ker,n,B)
  @distributed hcat for b in 1:B
    boot=map(x-> transpose(rand(x,n)),ker)
    cv=map(cov,boot)
    map((x,y,z)-> MixtureModel(map(u->MvNormal(u,x^2*y),[z[k,:] for k in 1:size(z)[1]])),bs,cv,boot)
  end
end

function boot_dens_opt(bs,ker,n,B)
  @distributed hcat for b in 1:B
    boot=map(x-> transpose(rand(x,n)),ker)
    cv=map((x,y)->[x^2*cov(y),y],bs,boot)
  end
end

function ecv_solver(N1,ker,pts,wts,np,lm,B,ecv)
  sizes=floor.(collect(range(N1/3,stop=N1*2,length=10)))
  ecvs=Array{Float64}(undef,length(sizes))
  for j in 1:length(sizes)
    neff=sizes[j]
    bs=map(x-> (4/(neff*(x+2)))^(1/(x+4)),np)
    dens= boot_dens(bs,ker,Int(round(sizes[j]/lm)),B)
    cvs=Array{Array}(undef,lm)
    for n in 1:lm
      cvs[n]= @distributed vcat for k in 1:size(pts[n])[2]
        vals= map(u->pdf(u,pts[n][:,k]),dens[n,:])
        std(vals)/mean(vals)
      end
    end
    s=0
    for n in 1:lm
      s=s+dot(cvs[n],wts[n].values)
    end
    ecvs[j]=s/sum(map(sum,wts))
  end
  model(x,p)=p[1]*x.^(-p[2])
  fit=curve_fit(model,sizes,ecvs,[0.5,0.5])
  Int(round((ecv/fit.param[1])^(1/(-fit.param[2]))))
end

function ecv_solver_opt(N1,ker,pts,wts,np,lm,B,ecv)
  sizes=floor.(collect(range(N1/3,stop=N1*2,length=10)))
  ecvs=Array{Float64}(undef,length(sizes))
  for j in 1:length(sizes)
    neff=sizes[j]
    bs=map(x-> (4/(neff*(x+2)))^(1/(x+4)),np)
    dens= boot_dens_opt(bs,ker,Int(round(sizes[j]/lm)),B)
    cvs=Array{Array}(lm)
    for n in 1:lm
      m=pmap(x->inv(x[1]),dens[n,:])
      d=pmap(x->(det((2*pi)^np[n]*x[1]))^(-0.5),dens[n,:])
      cvs[n]= @distributed vcat for k in 1:size(pts[n])[2]
        vals= map((x,y,z)->x*mean(map(a->exp(-0.5*transpose(pts[n][:,k]-a)*y*(pts[n][:,k]-a)),z[2])),d,m,dens[n,:])
        std(vals)/mean(vals)
      end
    end
    s=0
    for n in 1:lm
      s=s+dot(cvs[n],wts[n].values)
    end
    ecvs[j]=s/sum(map(sum,wts))
  end
  model(x,p)=p[1]*x.^(-p[2])
  fit=curve_fit(model,sizes,ecvs,[0.5,0.5])
  Int(round((ecv/fit.param[1])^(1/(-fit.param[2]))))
end


function APMC_KDE_adpt_init(N,expd,models,rho,;names=Vector[[string("parameter",i) for i in 1:length(models[m])] for m in 1:length(models)],prop=0.5,paccmin=0.02,ecv=0.1,B=10)
  i=1
  lm=length(models)
  N1=round(Int,N*prop)
  #array for number of parameters in each model
  np=Array{Int64}(undef,length(models))
  for j in 1:lm
    np[j]=length(models[j])
  end
  template=Array{Any}(undef,lm,1)
  #particles array
  pts=similar(template)
  #covariance matrix array
  sig=similar(template)
  #weights array
  wts=similar(template)
  #model probability at each iteration array
  p=zeros(lm,1)
  #array for KDE estimates
  ker=similar(template)
  temp=@distributed hcat for j in 1:N
    init(models,expd,np,rho)
  end
  its=[sum(temp[size(temp)[1],:])]
  epsilon=[quantile(collect(temp[maximum(np)+2,:]),prop)]
  pacc=ones(lm,1)
  println(round.([epsilon[i];its[i]],digits=3))
  temp=temp[:,findall(temp[maximum(np)+2,:].<=epsilon[i])]
  temp=temp[:,1:N1]
  for j in 1:lm
    pts[j,i]=temp[2:(np[j]+1),findall(temp[1,:].==j)]
    wts[j,i]=StatsBase.weights(fill(1.0,sum(temp[1,:].==j)))
  end
  dists=transpose(temp[(maximum(np)+2),:])
  for j in 1:lm
    p[j]=sum(wts[j,1])
  end
  for j in 1:lm
    sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
  end
  p=p./sum(p)
  nbs=Array{Integer}(undef,length(models))
  for j in 1:lm
    nbs[j]=length(wts[j,i])
    println(round.(hcat(mean(diag(sig[j,i])[1:(np[j])]),pacc[j,i],nbs[j],p[j,i]),digits=3))
  end
  for j in 1:lm
    mus=pts[j,i]
    neff=1/sum((wts[j,i].values./sum(wts[j,i])).^2)
    bs=(4/(neff*(np[j]+2)))^(1/(np[j]+4))
    ker[j,i]=MixtureModel(map(u->MvNormal(u,bs^2*sig[j,i]),[mus[:,i] for i in 1:size(mus)[2]]),wts[j,i].values/sum(wts[j,i]))
  end
  N2=ecv_solver(N1,ker[:,i],pts[:,i],wts[:,i],np,lm,B,ecv)
  while maximum(pacc[:,i])>paccmin
    pts=reshape(pts,i*length(models))
    sig=reshape(sig,i*length(models))
    wts=reshape(wts,i*length(models))
    ker=reshape(ker,i*length(models))
    for j in 1:length(models)
      push!(pts,Array{Any}(undef,1))
      push!(sig,Array{Any}(undef,1))
      push!(wts,Array{Any}(undef,1))
      push!(ker,Array{Any}(undef,1))
    end
    pts=reshape(pts,length(models),i+1)
    sig=reshape(sig,length(models),i+1)
    wts=reshape(wts,length(models),i+1)
    ker=reshape(ker,length(models),i+1)
    i=i+1
    if (ceil(N2/prop)-N1)>0
      temp2=@distributed hcat for j in 1:Int(ceil(N2/prop)-N1)
        cont_KDE(models,pts,wts,expd,np,i,ker,rho)
      end
      its=vcat(its,sum(temp2[size(temp2)[1],:]))
      temp=hcat(temp,temp2)
    else
      its=vcat(its,0)
      temp=temp
    end
    inds=sortperm(reshape(temp[maximum(np)+2,:],size(temp)[2]))[1:N2]
    temp=temp[:,inds]
    dists=hcat(dists,transpose(temp[(maximum(np)+2),:]))
    epsilon=vcat(epsilon,temp[(maximum(np)+2),N2])
    pacc=hcat(pacc,zeros(lm))
    for j in 1:lm
      if (ceil(N2/prop)-N1)>0
        pacc[j,i]=sum(temp[1,inds.>N1].==j)/max(sum(temp2[1,:].==j),0.000001)
      else
        pacc[j,i]=1
      end

    end
    println(round.(vcat(epsilon[i],its[i]),digits=3))
    for j in 1:lm
      pts[j,i]=temp[2:(np[j]+1),findall(temp[1,:].==j)]
      if size(pts[j,i])[2]>0
        keep=inds[findall(reshape(temp[1,:].==j,N2))].<=N1
        wts[j,i]= @distributed vcat for k in 1:length(keep)
          if !keep[k]
            pdf(models[j],(pts[j,i][:,k]))/pdf(ker[j,i-1],pts[j,i][:,k])
          else
            0.0
          end
        end
        l=1
        for k in 1:length(keep)
          if keep[k]
            wts[j,i][k]=wts[j,i-1][l]
            l=l+1
          end
        end
        wts[j,i]=StatsBase.weights(wts[j,i])
      else
        wts[j,i]=zeros(0)
      end
    end
    p=hcat(p,zeros(length(models)))
    for j in 1:lm
      p[j,i]=sum(wts[j,i])
    end
    for j in 1:lm
      if(size(pts[j,i])[2]>np[j])
        sig[j,i]=cov(pts[j,i],wts[j,i],2,corrected=false)
        #sig[j,i]=cov(transpose(pts[j,i]))
        if isposdef(sig[j,i])
          dker=MvNormal(pts[j,i-1][:,1],sig[j,i])
          if pdf(dker,pts[j,i][:,1])==Inf
            sig[j,i]=sig[j,i-1]
          end
        else
          sig[j,i]=sig[j,i-1]
        end
      else
        sig[j,i]=sig[j,i-1]
      end
    end
    p[:,i]=p[:,i]./sum(p[:,i])
    for j in 1:lm
      if(size(pts[j,i])[2]>0)
        mus=pts[j,i]
        neff=1/sum((wts[j,i].values./sum(wts[j,i])).^2)
        bs=(4/(neff*(np[j]+2)))^(1/(np[j]+4))
        ker[j,i]=MixtureModel(map(u->MvNormal(u,bs^2*sig[j,i]),[mus[:,i] for i in 1:size(mus)[2]]),wts[j,i].values/sum(wts[j,i]))
      else
        ker[j,i]= ker[j,i-1]
      end
    end
    for j in 1:lm
      nbs[j]=length(wts[j,i])
      println(round.(hcat(mean(diag(sig[j,i])./diag(sig[j,1])),pacc[j,i],nbs[j],p[j,i]),digits=3))
    end
    N1=N2
  end
  samp=ABCfit(pts,sig,wts,p,its,dists,epsilon,temp,pacc,names,models)
  return(samp)
end
