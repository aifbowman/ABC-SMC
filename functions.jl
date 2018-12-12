import StatsBase.mode
function mode(dens::UnivariateKDE)
    ind=findmax(dens.density)[2]
    dens.x[ind]
end

function mode_univar(samp)
  md=Array{Float64}(undef,size(samp)[1])
  for i in 1:size(samp)[1]
    dens=kde(samp[i,:])
    md[i]=mode(dens)
  end
  return(md)
end

function extract_ml_particle(fit::ABCfit,model)
  temp=fit.pts[model,end]
  md=mode_univar(temp)
  sepmat=broadcast(-,temp,md)
  for i in 1:size(sepmat)[1]
    sepmat[i,:]=sepmat[i,:]./std(sepmat[i,:])
  end
  dists=sqrt.(sum(sepmat.^2,dims=1))
  ind=findmin(dists)
  return((temp[:,ind[2][2]],ind[1]))
end

function intercept_cont(dat)
  fit=linreg(dat[:,2],dat[:,1])
  return(size(dat)[1]*fit[1]/(size(dat)[1]*fit[1]+sum(fit[2].*dat[:,2])))
end

function ms_ci(fit::ABCfit)
    p=fit.p
    nm=size(p)[1]
    out=Array{Float64}(nm)
    inds=fit.epsilon.<fit.epsilon[end]*4
    for i in 1:nm
        out[i]=std(p[i,inds])
    end
    return(out)
end

function ms_ci_boot(fit::ABCfit)
    nm=size(p)[1]
    out=Array{Float64}(nm)
    inds=fit.epsilon.<fit.epsilon[end]*4
    for i in 1:nm
        out[i]=std(p[i,inds])
    end
    return(out)
end

function boot_sub(d2,stat,r=1000)
  l=size(d2)[1]
  bootstrp=@distributed (x,y)->vcat(x,y) for i in 1:r
 d2[sample(1:size(d2)[1],size(d2)[1],replace=true),:]
end
stats=zeros(r)
for m in 1:r
  stats[m]=stat((bootstrp[l*(m-1)+1:m*l,:]))
end
return(stats)
end
function sliding_apply(a,w,f,d=size(a)[2])
  #a=a[sortperm(collect(a[:,d])),:]
  out=zeros(1000,2)
  l=minimum(a[:,d])
  u=maximum(a[:,d])
  k=(u-l-w)
  for i in 1:(size(out)[1])
    inds=(a[:,d].<l+w)&(a[:,d].>l)
    out[i,2]=mean(a[inds,d])
    out[i,1]=f(a[inds,:])
    l+=k/999
  end
  out
end
function sliding_apply_boot(a,w,f,d=size(a)[2],n=1000,ql=0.025,qu=0.975,l=minimum(a[:,d]),u=maximum(a[:,d]))
  #a=a[sortperm(collect(a[:,d])),:]
  out=zeros(1000,4)
  boots=zeros(n)
  k=(u-l-w)
  for j in 1:(size(out)[1])
    inds=find((a[:,d].<=l+w)&(a[:,d].>=l))
    if length(inds)>0
    out[j,1]=mean(a[inds,d])
    out[j,3]=f(a[inds,:])
    for i in 1:n
    temp=sample(inds,length(inds))
    boots[i]=f(a[temp,:])
  end
  out[j,2]=quantile(boots,ql)
  out[j,4]=quantile(boots,qu)
else
out[j,:]=out[j-1,:]
end
  l+=k/999
  end
  out
end
function sliding_apply_CI(a,w,f,d=size(a[1])[2],ql=0.025,qu=0.975)
  n=length(a)
  for i in 1:n
  a[i]=a[i][sortperm(collect(a[i][:,d])),:]
  end
  out=zeros(1000,4)
  l=minimum(a[1][:,d])
  u=maximum(a[1][:,d])
  boots=zeros(n)
  for j in 1:(size(out)[1])
    inds=find((a[1][:,d].<l+w)&(a[1][:,d].>l))
    out[j,1]=mean(a[1][inds,d])
    out[j,3]=f(a[1][inds,:])
    for i in 1:n
    boots[i]=f(a[i][inds,:])
  end
  out[j,2]=quantile(boots,ql)
  out[j,4]=quantile(boots,qu)
    l+=(u-l-w)/999
  end
  out
end
# function plot(x::ABCfit,models=1:size(x.pts)[1])
#   np=zeros(models[end])
#   for i in models
#     np[i]=size(x.pts[i,end])[1]
#   end
#   mw=maximum(np)
#   for i in models
#     for j in 1:np[i]
#       subplot(models[end],mw,(i-1)*mw+j)
#       plt[:hist](collect(x.pts[Int(i),end][Int(j),:]))
#     end
#   end
# end

@userplot InferPlot

@recipe function f(h::InferPlot)
     if !(typeof(h.args[1]) <: ABCfit)
       error("InferPlot requires an ABCfit")
     end
    x= h.args[1]
    models=1:size(x.pts)[1]
    np=Array{Int64}(models[end])
    for i in models
        np[i]=size(x.pts[i,end])[1]
      end
    mw=maximum(np)
    cols=distinguishable_colors(models[end]+1)[2:models[end]+1]
    #set up the subplots
    legend := false
    size:=(1920,1080)
    #link := :both
    #ticks := [nothing :auto nothing]
    grid := false
    #  margin --> 0mm
    #  bottom_margin--> 0mm
    #  right_margin--> 0mm
    #  left_margin--> 0mm
    #  top_margin--> 0mm
    #foreground_color_subplot := [RGBA(0,0,0,0) :match RGBA(0,0,0,0)]
    layout:= @layout [a{0.1h}; grid(mw,models[end]*mw)]
    inds=reshape(1:mw^2*models[end],mw*models[end],mw)+1
    @series begin
        seriestype:= :bar
        xlabel :="model number"
        ylabel :="Posterior Probability"
        c:=cols
        subplot=1
        x.p[:,end]
    end
    for m in models
        for i in 1:np[m]
            @series begin
                seriestype := :histogram
                normed := true
                c:=cols[m]
                subplot := inds[mw*(m-1)+i,i]
                x.pts[m,end][i,:]
            end
            @series begin
                seriestype := :line
                c:=cols[m]
                subplot := inds[mw*(m-1)+i,i]
                xlabel :=x.names[m][i]
                ylabel :="Density"
                annotate:=(median(x.models[m][i]),pdf(x.models[m][i],maximum(x.models[m][i])),text(string("KS=",round(ExactOneSampleKSTest(x.pts[m,end][i,:],x.models[m][i]).δ,3)),10))
                x.models[m][i]
            end
            for j in setdiff(1:np[m],i)
                @series begin
                    if i<j
                    seriestype := :scatter
                    markercolor:=cols[m]
                    linecolor:= :black
                    smooth := true
                    markerstrokewidth --> 0
                    annotate:=(mean(x.pts[m,end][i,:]),minimum(x.pts[m,end][j,:]),text(string("RHO=",round(cor(x.pts[m,end][i,:],x.pts[m,end][j,:]),3)),10))
                else
                     seriestype := :histogram2d
                 end
                    subplot := inds[mw*(m-1)+i,j]
                    xlabel :=x.names[m][i]
                    ylabel :=x.names[m][j]
                    x.pts[m,end][i,:], x.pts[m,end][j,:]
                end
            end
            for j in setdiff(1:mw,1:np[m])
                @series begin
                subplot := inds[mw*(m-1)+i,j]
                seriestype:= :line
                border:=false
                ticks:=false
                linealpha:=0
                1:10, 1:10
            end
            end
        end
        for i in setdiff(1:mw,1:np[m])
            for j in 1:mw
            @series begin
            subplot := inds[mw*(m-1)+i,j]
            seriestype:= :line
            border:=false
            ticks:=false
            linealpha:=0
            1:10, 1:10
        end
    end
end
end
end


@recipe function f(h::InferPlot)
     if !(typeof(h.args[1]) <: ABCfit)
       error("InferPlot requires an ABCfit")
     end
    x= h.args[1]
    models=1:size(x.pts)[1]
    np=Array{Int64}(models[end])
    for i in models
        np[i]=size(x.pts[i,end])[1]
      end
    mw=maximum(np)
    #set up the subplots
    legend := false
    size:=(3440,1440)
    #link := :both
    #ticks := [nothing :auto nothing]
    grid := false
     margin --> 0mm
     bottom_margin--> 0mm
     right_margin--> 0mm
     left_margin--> 0mm
     top_margin--> 0mm
    #foreground_color_subplot := [RGBA(0,0,0,0) :match RGBA(0,0,0,0)]
    layout :=@layout [grid(i,i) for j=1, i in np]
    ind=1
    for m in models
        for i in 1:np[m]
            for j in 1:np[m]
                if i==j
            @series begin
                seriestype := :histogram
                normed := true
                subplot := ind
                x.pts[m,end][i,:]
            end
            @series begin
                seriestype := :line
                subplot := ind
                xlabel :=x.names[m][i]
                ylabel :="Density"
                annotate:=(median(x.models[m][i]),pdf(x.models[m][i],maximum(x.models[m][i])),text(string("KS=",round(ExactOneSampleKSTest(x.pts[m,end][i,:],x.models[m][i]).δ,3)),10))
                x.models[m][i]
            end
        else
                @series begin
                    if i<j
                    seriestype := :scatter
                    smooth := true
                    markerstrokewidth --> 0
                    annotate:=(mean(x.pts[m,end][i,:]),minimum(x.pts[m,end][j,:]),text(string("RHO=",round(cor(x.pts[m,end][i,:],x.pts[m,end][j,:]),3)),10))
                else
                     seriestype := :histogram2d
                 end
                    subplot := ind
                    xlabel :=x.names[m][i]
                    ylabel :=x.names[m][j]
                    x.pts[m,end][i,:], x.pts[m,end][j,:]
                end
                ind+=1
            end
        end
    end
end
end

@recipe function f(h::InferPlot)
     if !(typeof(h.args[1]) <: ABCfit)
       error("InferPlot requires an ABCfit")
     end
    x= h.args[1]
    models=1:size(x.pts)[1]
    np=Array{Int64}(models[end])
    for i in models
        np[i]=size(x.pts[i,end])[1]
      end
    mw=maximum(np)
    #set up the subplots
    legend := false
    size:=(1920,1080)
    #link := :both
    #ticks := [nothing :auto nothing]
    grid := false
     margin --> 0mm
     bottom_margin--> 0mm
     right_margin--> 0mm
     left_margin--> 0mm
     top_margin--> 0mm
    #foreground_color_subplot := [RGBA(0,0,0,0) :match RGBA(0,0,0,0)]
    for m in models
        reuse=false
        layout :=(np[m],np[m])
        inds=reshape(1:np[m]^2,np[m],np[m])
        for i in 1:np[m]
            @series begin
                seriestype := :histogram
                normed := true
                subplot := inds[i,i]
                x.pts[m,end][i,:]
            end
            @series begin
                seriestype := :line
                subplot := inds[i,i]
                xlabel :=x.names[m][i]
                ylabel :="Density"
                annotate:=(median(x.models[m][i]),pdf(x.models[m][i],maximum(x.models[m][i])),text(string("KS=",round(ExactOneSampleKSTest(x.pts[m,end][i,:],x.models[m][i]).δ,3)),10))
                x.models[m][i]
            end
            for j in setdiff(1:np[m],i)
                @series begin
                    if i<j
                    seriestype := :scatter
                    smooth := true
                    markerstrokewidth --> 0
                    annotate:=(mean(x.pts[m,end][i,:]),minimum(x.pts[m,end][j,:]),text(string("RHO=",round(cor(x.pts[m,end][i,:],x.pts[m,end][j,:]),3)),10))
                else
                     seriestype := :histogram2d
                 end
                    subplot := inds[i,j]
                    xlabel :=x.names[m][i]
                    ylabel :=x.names[m][j]
                    x.pts[m,end][i,:], x.pts[m,end][j,:]
                end
            end
        end
    end
end

@recipe f(::Type{ABCfit}, x::ABCfit) = transpose(x.pts[1,end])

@recipe f(::Type{KernelDensity.UnivariateKDE}, y::KernelDensity.UnivariateKDE) = y.x, y.density

import Distributions.rand
import Distributions.pdf

function rand(x::Vector)
  y=zeros(length(x))
  for i in 1:length(x)
    y[i]=rand(x[i])
  end
  return(y)
end

function pdf(x::Vector,z::Vector)
  y=Vector(undef,length(x))
  for i in 1:length(x)
    y[i]=pdf(x[i],z[i])
  end
  return(prod(y))
end

# function plot(d::KernelDensity.UnivariateKDE)
#     plot(d.x,d.density)
# end
