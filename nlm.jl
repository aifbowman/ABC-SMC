using Distributions
using KernelDensity
function cycl(a,b,eps1,eps2,colocs,tf,div,start=6,t0=0,nm=1,mrnas=false)
  d1=Distributions.Normal(0.0,eps1)
  d2=Distributions.Normal(0.5,eps2)
  out=zeros(div,3+nm)
  fin=max((a*start+b+rand(d1,1)[1]),start)
  out[:,1]=(log(2,fin)-log(2,start)).*(0:(div-1))/(div)+t0
  out[:,2]=start.*(2 .^(out[:,1]-fill(t0,div)))
  if mrnas
    out[:,3]=fill(min(max(rand(d2,1)[1],0.0001),1),div)
    sigmat=colocs*out[1,3]*(1-out[1,3])
    d3=Distributions.MvNormal(fill(0.0,nm),sigmat)
    out[:,4:(3+nm)]=repmat(rand(d3,1),div,1)
    return(Any[out,out[1,3],fin,t0+log(2,fin/start)])
  end
  return(Any[out[:,1:2],min(max(rand(d2,1)[1],0.0001),1),fin,t0+log(2,fin/start)])
end

function nlm_lin(a,b,eps1,eps2,tf,div,start=6,nm=1,colocs=0,mrnas=false)
  tf=ceil(tf)
  if colocs==0
    colocs=diagm(nm)
  end
  if mrnas
    out=zeros(tf*div,3+nm)
  else
    out=zeros(tf*div,2)
  end
  temp=Array(Any,4)
  temp[2]=start
  temp[3]=1
  temp[4]=0
  for i in 1:tf
    temp=cycl(a,b,eps1,eps2,colocs,tf,div,temp[2]*temp[3],temp[4],nm)
    out[((div*(i-1)+1):(div*i)),:]=temp[1]
  end
  return(out)
end

function nlm_sampler(x,tf,div,start=6,bias=false)
  raw=nlm(x[1],x[2],x[3],x[4],(tf+100),div,start)
  starts=raw[(100*div+1):((100+tf)*div),1][1:div:length(raw[(100*div+1):((100+tf)*div),1])]
  ends=raw[(100*div+1):((100+tf)*div),1][99:div:length(raw[(100*div+1):((100+tf)*div),1])]
  times=ends-starts
  if bias
    inds=div*wsample(0:(tf-1),times,tf)+sample(1:div,tf)+100*div
    lens=raw[inds,2]
  else
    #     wts=zeros(div*tf)
    #     for i in 1:tf
    #     wts[(div*(i-1)+1):(div*i)]=fill(times[i],div).*2 .^[0:-1/(div-1):-1]
    #   end
    #     lens=wsample(raw[(100*div+1):((100+tf)*div),2],wts,tf)
    inds=div*wsample(0:(tf-1),times,tf)+wsample(1:div,2 .^collect(0:-1/(div-1):-1),tf)+100*div
    lens=raw[inds,2]
  end
end


function dts(a,b,eps1,eps2,tf,div,output)
  Any[cycl(a,b,eps1,eps2,diagm(1),tf,div,output[3]*output[2],output[4]),cycl(a,b,eps1,eps2,diagm(1),tf,div,output[3]*(1-output[2]),output[4])]
end


function nlm2(a,b,eps1,eps2,tf,div,start=6,col=false)
  temp=cycl(a,b,eps1,eps2,diagm(1),tf,div,start,0)
  colony=Array(Any,2^ceil(tf)-1)
  colony[1]=temp
  temp=Any[temp]
  for i in 2:ceil(tf)
    temp2=Array(Any,2^((i-1)))
    for j in 1:2^(i-2)
      temp2[(2*(j-1)+1):(2*j)]=dts(a,b,eps1,eps2,tf,div,temp[j])
    end
    temp=temp2
    colony[(2^(i-1)):(2^(i)-1)]=temp
    if (!col)&(i>2)
      colony[(2^(i-3)):(2^(i-2)-1)]=fill(0,2^(i-3))
    end
  end
  #   starts=Array{Float64}(2^(ceil(tf)-2))
  #   for i in (length(colony)-(2^(ceil(tf)-1)+2^(ceil(tf)-2))+1):(length(colony)-(2^(ceil(tf)-1)))
  #     starts[i-(length(colony)-(2^(ceil(tf)-1)+2^(ceil(tf)-2)))]=colony[i][4]
  #   end
  #   fins=Array{Float64}(2^(ceil(tf)-1))
  #   for i in (length(colony)-2^(ceil(tf)-1)+1):(length(colony))
  #     fins[i-(length(colony)-2^(ceil(tf)-1))]=colony[i][4] b
  #   end
  #   start=minimum(starts)
  #   fin=minimum(fins)
  #   println([start,fin])
  #   d=Uniform(start,fin)
  if col
    return(colony)
  end
  d=Uniform(tf-2,tf-1)
  t=rand(d,1)[1]
  samp=Array{Float64}(undef,length(colony))
  for i in 1:length(colony)
    if(colony[i]!=0)
      if (colony[i][4]>t)&(colony[i][1][1,1]<t)
        samp[i]=colony[i][1][findmin(abs(colony[i][1][:,1]-t))[2],2]
      else
        samp[i]=0
      end
    end
  end
  samp=samp[samp.>0]
  return(samp)
end

# function nlm_init(params,N)
#     a=params[1]
#     b=params[2]
#     eps1=params[3]
#     eps2=params[4]
#     eps3=params[5]
#     raw=Array{Float64}(undef,N)
#     d1=Normal(0.0,eps1)
#     d2=Normal(0.5,eps2)
#     if eps3!=0
#       d3=TruncatedNormal(1.0,eps3,0.0,Inf)
#     end
#     starts=fill(b/(2-a),N)
#     fins=max.(a*starts.+b.+rand(d1,N),starts)
#     t0=rand(Uniform(0,1),N)
#     raw[:]=starts.+(fins.-starts).*t0
#     if eps3==0
#       grs=fill(1,N)
#     else
#       grs=rand(d3,N)
#     end
#     div_waits=log.(2,fins./starts).*(1 .-t0)./grs
#     return starts, fins, raw, grs, div_waits
# end

function nlm_init(params,N)
    a=params[1]
    b=params[2]
    eps1=params[3]
    eps2=params[4]
    eps3=params[5]
    raw=Array{Float64}(undef,N)
    d1=Normal(0.0,eps1)
    d2=Normal(0.5,eps2)
    if eps3!=0
      d3=TruncatedNormal(1.0,eps3,0.0,Inf)
    end
    d4=Normal(0.0,eps1/(1-(a^2)/4))
    starts=max.(rand(d2,N).*(fill(2*b/(2-a),N).+rand(d4,N)),fill(0.0,N))
    fins=max.(a*starts.+b.+rand(d1,N),starts)
    t0=2 .^rand(Uniform(0,1),N).-fill(1.0,N)
    raw[:]=starts.+(fins.-starts).*t0
    if eps3==0
      grs=fill(1,N)
    else
      grs=rand(d3,N)
    end
    div_waits=log.(2,fins./starts).*(1 .-t0)./grs
    return starts, fins, raw, grs, div_waits
end

function nlm_div!(i,t,dt,raw,div_waits,starts,fins,grs,gf,params,N,unbiased)
  a=params[1]
  b=params[2]
  eps1=params[3]
  eps2=params[4]
  eps3=params[5]
  d1=Normal(0.0,eps1)
  d2=Normal(0.5,eps2)
  if eps3!=0
    d3=TruncatedNormal(1.0,eps3,0.0,Inf)
  end
  div=min(max(rand(d2),0),1)
  if unbiased
    dtr=sample(1:(N+1))
  else
    dtr=i
  end
  if dtr<(N+1)
    starts[dtr]=fins[i]*(1-div)
    if eps3!=0
      grs[dtr]=rand(d3)
      gf[dtr]=2^(grs[dtr]*dt)
    end
    fins[dtr]=max(a*starts[dtr]+b+rand(d1),starts[dtr])
    raw[dtr]=starts[dtr]*2^(-grs[dtr]*div_waits[i])
    div_waits[dtr]=div_waits[i]+log(2,fins[dtr]/starts[dtr])/grs[dtr]
  end
  if dtr!=i
    starts[i]=fins[i]*div
    if eps3!=0
      grs[i]=rand(d3)
      gf[i]=2^(grs[i]*dt)
    end
    raw[i]=starts[i]*2^(-grs[i]*div_waits[i])
    fins[i]=max(a*starts[i]+b+rand(d1),starts[i])
    div_waits[i]=div_waits[i]+log(2,fins[i]/starts[i])/grs[i]
  end
  return(div)
end

function nlm(params,N,tf,dt,unbiased=true,times=false)
  starts, fins, raw, grs, div_waits =nlm_init(params,N)
  gf=2 .^(grs.*dt)
  t=0
  while t<tf
    t=t+dt
    for i in 1:N
      div_waits[i]=div_waits[i]-dt
      if div_waits[i]<0
        nlm_div!(i,t,dt,raw,div_waits,starts,fins,grs,gf,params,N,unbiased)
      else
        raw[i]=min(raw[i]*gf[i],fins[i])
      end
    end
  end
  if times
    return hcat(starts,fins,div_waits,grs,raw)
  else
    return raw
  end
end

function nlm_conv(params,N,tf,dt,unbiased=true)
  starts, fins, raw, grs, div_waits =nlm_init(params,N)
  prev=raw[:]
  curr=raw[:]
  ks=Array{Float64}(undef,Int(ceil(tf/dt))+1)
  gf=2 .^(grs.*dt)
  t=0
  i=0
  while t<tf
    t=t+dt
    i=i+1
    for i in 1:N
      div_waits[i]=div_waits[i]-dt
      if div_waits[i]<0
        nlm_div!(i,t,dt,raw,div_waits,starts,fins,grs,gf,params,N,unbiased)
      else
        raw[i]=min(raw[i]*gf[i],fins[i])
      end
    end
    prev[:]=curr[:]
    curr[:]=raw[:]
    ks[i]=ApproximateTwoSampleKSTest(prev,curr).δ
  end
  return(ks)
end

function nlm_NE(params,N,tf,dt,unbiased=true,times=false)
    a=params[1]
    b=params[2]
    t_NE=params[3]
    eps1=params[4]
    eps2=params[5]
    eps3=params[6]
    d1=Normal(0.0,eps1)
    d2=Normal(0.5,eps2)
    if eps3!=0
        d3=TruncatedNormal(1.0,eps3,0.0,Inf)
    end
    # for i in 1:N
    #     if t0[i]>1/(1+t_NE)
    #         raw[i]=fins[i]
    #     else
    #         raw[i]=starts[i]+(fins[i]-starts[i]).*t0[i]*(1+t_NE)
    #     end
    # end
    starts, fins, raw, grs, div_waits =nlm_init(params[vcat(1:2,4:6)],N)
    gf=2 .^(grs.*dt)
    t=0
    while t<tf
        t=t+dt
        for i in 1:N
            div_waits[i]=div_waits[i]-dt
            if div_waits[i]<0
                div=min(max(rand(d2),0),1)
                if unbiased
                    dtr=sample(1:(N+1))
                else
                    dtr=i
                end
                if dtr<(N+1)
                    starts[dtr]=fins[i]*(1-div)
                    if eps3!=0
                        grs[dtr]=rand(d3)
                    end
                    gf[dtr]=2^(grs[dtr]*dt/t_NE)
                    fins[dtr]=max(a*starts[dtr]+b+rand(d1),starts[dtr])
                    raw[dtr]=starts[dtr]*2^(-grs[dtr]*div_waits[i])
                    div_waits[dtr]=div_waits[i]+log(2,fins[dtr]/starts[dtr])/grs[dtr]
                end
                if dtr!=i
                    starts[i]=fins[i]*div
                    if eps3!=0
                        grs[i]=rand(d3)
                    end
                    gf[i]=2^(grs[i]*dt/t_NE)
                    raw[i]=starts[i]*2^(-grs[i]*div_waits[i])
                    fins[i]=max(a*starts[i]+b+rand(d1),starts[i])
                    div_waits[i]=div_waits[i]+log(2,fins[i]/starts[i])/grs[i]
                end
            else
                if raw[i]<fins[i]
                    raw[i]=min(raw[i]*gf[i],fins[i])
                end
            end
        end
    end
    if times
        return hcat(starts,fins,div_waits,grs,raw)
    else
        return raw
    end
end

function nlm_NE_full(params,N,tf,dt,unbiased=true,times=false)
    a=params[1]
    b=params[2]
    t_NE_on=params[3]
    t_NE_off=params[4]
    if t_NE_off<t_NE_on
        return(fill(0.0,N))
    end
    gm=params[5]
    eps1=params[6]
    eps2=params[7]
    eps3=params[8]
    d1=Normal(0.0,eps1)
    d2=Normal(0.5,eps2)
    if eps3!=0
        d3=TruncatedNormal(1.0,eps3,0.0,Inf)
    end
    # for i in 1:N
    #     if t0[i]>1/(1+t_NE)
    #         raw[i]=fins[i]
    #     else
    #         raw[i]=starts[i]+(fins[i]-starts[i]).*t0[i]*(1+t_NE)
    #     end
    # end
    starts, fins, raw, grs, div_waits =nlm_init(params[vcat(1:2,6:8)],N)
    gf=hcat(2 .^(grs.*dt),2 .^(gm*grs.*dt))
    k=fill(0.0,N)
    t=0
    while t<tf
        t=t+dt
        for i in 1:N
            div_waits[i]=div_waits[i]-dt
            if div_waits[i]<0
                div=min(max(rand(d2),0),1)
                if unbiased
                    dtr=sample(1:(N+1))
                else
                    dtr=i
                end
                if dtr<(N+1)
                    starts[dtr]=fins[i]*(1-div)
                    if eps3!=0
                        grs[dtr]=rand(d3)
                    end
                    gf[dtr,1]=2^(grs[dtr]*dt/(t_NE_on+gm-gm*t_NE_off))
                    gf[dtr,2]=2^(gm*grs[dtr]*dt/(t_NE_on+gm-gm*t_NE_off))
                    fins[dtr]=max(a*starts[dtr]+b+rand(d1),starts[dtr])
                    raw[dtr]=starts[dtr]*2^(-grs[dtr]*div_waits[i])
                    k[dtr]=log(2,fins[dtr]/starts[dtr])/grs[dtr]
                    div_waits[dtr]=div_waits[i]+k[dtr]
                end
                if dtr!=i
                    starts[i]=fins[i]*div
                    if eps3!=0
                        grs[i]=rand(d3)
                    end
                    gf[i,1]=2^(grs[i]*dt/(t_NE_on+gm-gm*t_NE_off))
                    gf[i,2]=2^(gm*grs[i]*dt/(t_NE_on+gm-gm*t_NE_off))
                    raw[i]=starts[i]*2^(-grs[i]*div_waits[i])
                    fins[i]=max(a*starts[i]+b+rand(d1),starts[i])
                    k[i]=log(2,fins[i]/starts[i])/grs[i]
                    div_waits[i]=div_waits[i]+k[i]
                end
            else
                if raw[i]<fins[i]
                    if (div_waits[i]+dt)<k[i]*(1-t_NE_off)
                        raw[i]=min(raw[i]*gf[i,2],fins[i])
                    elseif (div_waits[i]+dt)>k[i]*(1-t_NE_on)
                        raw[i]=min(raw[i]*gf[i,1],fins[i])
                    end
                end
            end
        end
    end
    if times
        return hcat(starts,fins,div_waits,grs,raw)
    else
        return raw
    end
end

function nlm_SE(params,N,tf,dt,unbiased=true,times=false)
    a=params[1]
    b=params[2]
    t_SE=params[3]
    gm=params[4]
    eps1=params[5]
    eps2=params[6]
    eps3=params[7]
    d1=Normal(0.0,eps1)
    d2=Normal(0.5,eps2)
    if eps3!=0
        d3=TruncatedNormal(1.0,eps3,0.0,Inf)
    end
    # for i in 1:N
    #     if t0[i]>1/(1+t_NE)
    #         raw[i]=fins[i]
    #     else
    #         raw[i]=starts[i]+(fins[i]-starts[i]).*t0[i]*(1+t_NE)
    #     end
    # end
    starts, fins, raw, grs, div_waits =nlm_init(params[vcat(1:2,5:7)],N)
    gf=hcat(2 .^(grs.*dt),2 .^(gm*grs.*dt))
    k=fill(0.0,N)
    t=0
    while t<tf
        t=t+dt
        for i in 1:N
            div_waits[i]=div_waits[i]-dt
            if div_waits[i]<0
                div=min(max(rand(d2),0),1)
                if unbiased
                    dtr=sample(1:(N+1))
                else
                    dtr=i
                end
                if dtr<(N+1)
                    starts[dtr]=fins[i]*(1-div)
                    if eps3!=0
                        grs[dtr]=rand(d3)
                    end
                    gf[dtr,1]=2^(grs[dtr]*dt/(t_SE+gm-gm*t_SE))
                    gf[dtr,2]=2^(gm*grs[dtr]*dt/(t_SE+gm-gm*t_SE))
                    fins[dtr]=max(a*starts[dtr]+b+rand(d1),starts[dtr])
                    raw[dtr]=starts[dtr]*2^(-grs[dtr]*div_waits[i])
                    k[dtr]=log(2,fins[dtr]/starts[dtr])/grs[dtr]
                    div_waits[dtr]=div_waits[i]+k[dtr]
                end
                if dtr!=i
                    starts[i]=fins[i]*div
                    if eps3!=0
                        grs[i]=rand(d3)
                    end
                    gf[i,1]=2^(grs[i]*dt/(t_SE+gm-gm*t_SE))
                    gf[i,2]=2^(gm*grs[i]*dt/(t_SE+gm-gm*t_SE))
                    raw[i]=starts[i]*2^(-grs[i]*div_waits[i])
                    fins[i]=max(a*starts[i]+b+rand(d1),starts[i])
                    k[i]=log(2,fins[i]/starts[i])/grs[i]
                    div_waits[i]=div_waits[i]+k[i]
                end
            else
                if raw[i]<fins[i]
                    if (div_waits[i]+dt)<k[i]*(1-t_SE)
                        raw[i]=min(raw[i]*gf[i,2],fins[i])
                    else
                        raw[i]=min(raw[i]*gf[i,1],fins[i])
                    end
                end
            end
        end
    end
    if times
        return hcat(starts,fins,div_waits,grs,raw)
    else
        return raw
    end
end

function nlm_nuc(params,N,tf,dt,unbiased=true,times=false)
  a=reshape(params[1:2],1,2)
  b=reshape(params[3:4],1,2)
  eps1=params[5:6].^2
  eps2=params[7]
  eps3=params[8]
  eps4=params[9]
  eps5=params[10]
  gf=fill(2^dt,N,2)
  d1=MvNormal(fill(0.0,2),eps1)
  d2=MvNormal(fill(0.5,2),[eps2^2 eps4*eps2*eps3;eps4*eps2*eps3 eps3^2])
  if eps5!=0
    d3=TruncatedNormal(1.0,eps5,0.0,Inf)
  end
  #init=Uniform(0.5*b/(2-a),2*b/(2-a))
  #starts=rand(init,N)
  starts=repmat(b./(2-a),N)
  fins=max(a.*starts.+b+transpose(rand(d1,N)),starts)
  t0=rand(Uniform(0,1),N)
  raw=starts+(fins-starts).*hcat(t0,t0)
  if eps5==0
    grs=fill(1,N)
  else
    grs=rand(d3,N)
  end
  div_waits=log(2,fins[:,1]./starts[:,1]).*(1-t0)./grs
  t=0
  while t<tf
    t=t+dt
    for i in 1:N
      div_waits[i]=div_waits[i]-dt
      if(div_waits[i]<0)
        div=min(max(reshape(rand(d2),1,2),0),1)
        if unbiased
          dtr=sample(1:(N+1))
        else
          dtr=i
        end
        if dtr<(N+1)
          for k in 1:2
            starts[dtr,k]=fins[i,k]*(1-div[k])
          end
          if eps5!=0
            grs[dtr]=rand(d3)
            gf[dtr,1]=2^(grs[dtr]*dt)
          end
          noise=rand(d1)
          for k in 1:2
            fins[dtr,k]=max(a[k]*starts[dtr,k]+b[k]+noise[k],starts[dtr,k])
          end
          gf[dtr,2]=(fins[dtr,2]/starts[dtr,2])^(dt/(log(2,fins[dtr,1]/starts[dtr,1])/grs[dtr]))
          raw[dtr,:]=starts[dtr,:].*vcat(2^(-grs[dtr]*div_waits[i]),(fins[dtr,2]/starts[dtr,2])^(-div_waits[i]/(log(2,fins[dtr,1]/starts[dtr,1])/grs[dtr])))
          div_waits[dtr]=div_waits[i]+log(2,fins[dtr,1]/starts[dtr,1])/grs[dtr]
        end
        if dtr!=i
          for k in 1:2
            starts[i,k]=fins[i,k]*(div[k])
          end
          if eps5!=0
            grs[i]=rand(d3)
            gf[i,1]=2^(grs[i]*dt)
          end
          noise=rand(d1)
          for k in 1:2
            fins[i,k]=max(a[k]*starts[i,k]+b[k]+noise[k],starts[i,k])
          end
          gf[i,2]=(fins[i,2]/starts[i,2])^(dt/(log(2,fins[i,1]/starts[i,1])/grs[i]))
          raw[i,1]=starts[i,1]*2^(-grs[i]*div_waits[i])
          raw[i,2]=starts[i,2]*(fins[i,2]/starts[i,2])^(-div_waits[i]/(log(2,fins[i,1]/starts[i,1])/grs[i]))
          div_waits[i]=div_waits[i]+log(2,fins[i,1]/starts[i,1])/grs[i]
        end
      else
        for k in 1:2
          raw[i,k]=raw[i,k]*gf[i,k]
        end
      end
    end
  end
  if times
    return hcat(starts,fins,div_waits,grs,raw)
  else
    return raw
  end
end

function nlm_bilinear(params,N,tf,dt,unbiased=true,times=false)
  a=reshape(params[1:4],2,2)
  b=reshape(params[5:6],1,2)
  gf=fill(2^dt,N,2)
  d1=MvNormal(fill(0.0,2),[params[7]^2 params[7]*params[8]*params[9];params[7]*params[8]*params[9] params[9]^2])
  d2=MvNormal(fill(0.5,2),[params[10]^2 params[10]*params[11]*params[12];params[10]*params[11]*params[12] params[12]^2])
  if params[13]!=0
    d3=TruncatedNormal(1.0,params[13],0.0,Inf)
  end
  #init=Uniform(0.5*b/(2-a),2*b/(2-a))
  #starts=rand(init,N)
  starts=repmat([b[1]/(2-a[1,1]) b[2]/(2-a[2,2])],N,1)
  fins=max.(starts*a+repmat(b,N,1)+transpose(rand(d1,N)),starts)
  t0=rand(Uniform(0,1),N)
  raw=starts+(fins-starts).*hcat(t0,t0)
  if params[13]==0
    grs=fill(1,N)
  else
    grs=rand(d3,N)
  end
  div_waits=log.(2,fins[:,1]./starts[:,1]).*(1-t0)./grs
  t=0
  while t<tf
    t=t+dt
    for i in 1:N
      div_waits[i]=div_waits[i]-dt
      if(div_waits[i]<0)
        div=min.(max.(reshape(rand(d2),1,2),0),1)
        if unbiased
          dtr=sample(1:(N+1))
        else
          dtr=i
        end
        if dtr<(N+1)
          for k in 1:2
            starts[dtr,k]=fins[i,k]*(1-div[k])
          end
          if params[13]!=0
            grs[dtr]=rand(d3)
            gf[dtr,1]=2^(grs[dtr]*dt)
          end
          noise=rand(d1)
          for k in 1:2
            fins[dtr,k]=max(a[k]*starts[dtr,k]+b[k]+noise[k],starts[dtr,k])
          end
          gf[dtr,2]=(fins[dtr,2]/starts[dtr,2])^(dt/(log(2,fins[dtr,1]/starts[dtr,1])/grs[dtr]))
          raw[dtr,:]=starts[dtr,:].*vcat(2^(-grs[dtr]*div_waits[i]),(fins[dtr,2]/starts[dtr,2])^(-div_waits[i]/(log(2,fins[dtr,1]/starts[dtr,1])/grs[dtr])))
          div_waits[dtr]=div_waits[i]+log(2,fins[dtr,1]/starts[dtr,1])/grs[dtr]
        end
        if dtr!=i
          for k in 1:2
            starts[i,k]=fins[i,k]*(div[k])
          end
          if params[13]!=0
            grs[i]=rand(d3)
            gf[i,1]=2^(grs[i]*dt)
          end
          noise=rand(d1)
          for k in 1:2
            fins[i,k]=max(a[k]*starts[i,k]+b[k]+noise[k],starts[i,k])
          end
          gf[i,2]=(fins[i,2]/starts[i,2])^(dt/(log(2,fins[i,1]/starts[i,1])/grs[i]))
          raw[i,1]=starts[i,1]*2^(-grs[i]*div_waits[i])
          raw[i,2]=starts[i,2]*(fins[i,2]/starts[i,2])^(-div_waits[i]/(log(2,fins[i,1]/starts[i,1])/grs[i]))
          div_waits[i]=div_waits[i]+log(2,fins[i,1]/starts[i,1])/grs[i]
        end
      else
        for k in 1:2
          raw[i,k]=raw[i,k]*gf[i,k]
        end
      end
    end
  end
  if times
    return hcat(starts,fins,div_waits,grs,raw)
  else
    return raw
  end
end

function nlm_stat(params,N,tf,dt,unbiased=true,times=false)
  a=params[1]
  b=params[2]
  eps1=params[3]
  eps2=params[4]
  eps3=params[5]
  c=params[7]
  d=params[6]
  eps4=params[8]
  gf=fill(2^dt,N)
  raw=Array{Float64}(undef,N)
  d1=Normal(0.0,eps1)
  d2=Normal(0.5,eps2)
  if eps3!=0
    d3=TruncatedNormal(1.0,eps3,0.0,Inf)
  end
  d4=TruncatedNormal(c,eps4,0,1)
  init=Uniform(0.5*b/(2-a),2*b/(2-a))
  starts=rand(init,N)
  fins=max(a*starts+b+rand(d1,N),starts)
  t0=rand(Uniform(0,1),N)
  raw[:]=starts+(fins-starts).*t0
  if eps3==0
    grs=fill(1,N)
  else
    grs=rand(d3,N)
  end
  div_waits=log(2,fins./starts).*(1-t0)./grs
  sep_time=zeros(N,2)
  t=0
  while t<tf
    t=t+dt
    for i in 1:N
      div_waits[i]=div_waits[i]-dt
      for j in 1:2
        sep_time[i,j]=sep_time[i,j]-dt
      end
      if(div_waits[i]<0)
        div=min(max(rand(d2),0),1)
        if unbiased
          dtr=sample(1:(N+1))
        else
          dtr=i
        end
        if dtr<(N+1)
          starts[dtr]=fins[i]*(1-div)
          if eps3!=0
            grs[dtr]=rand(d3)
            gf[dtr]=2^(grs[dtr]*dt)
          end
          fins[dtr]=max(a*starts[dtr]+b+rand(d1),starts[dtr])
          raw[dtr]=starts[dtr]*2^(-grs[dtr]*div_waits[i])
          sep=rand(d4)
          div_waits[dtr]=(div_waits[i]+log(2,fins[dtr]/starts[dtr])/grs[dtr])*(1+sep)
          sep_time[dtr,1]=d*div_waits[dtr]/(1+sep)
          sep_time[dtr,2]=d*div_waits[dtr]/(1+sep)+sep*div_waits[dtr]/(1+sep)

        end
        if dtr!=i
          starts[i]=fins[i]*div
          if eps3!=0
            grs[i]=rand(d3)
            gf[i]=2^(grs[i]*dt)
          end
          raw[i]=starts[i]*2^(-grs[i]*div_waits[i])
          fins[i]=max(a*starts[i]+b+rand(d1),starts[i])
          sep=rand(d4)
          div_waits[i]=(div_waits[i]+log(2,fins[i]/starts[i])/grs[i])*(1+sep)
          sep_time[i,1]=d*div_waits[i]/(1+sep)
          sep_time[i,2]=d*div_waits[i]/(1+sep)+sep*div_waits[i]/(1+sep)
        end
      else
        if((sep_time[i,1]>=0)||(sep_time[i,2]<=0))
          raw[i]=raw[i]*gf[i]
        end
      end
    end
  end
  if times
    return hcat(starts,fins,div_waits,grs,raw)
  else
    return raw
  end
end


function nlm_ext(params,N,tf,dt,unbiased)
  a=params[1]
  b=params[2]
  eps1=params[3]
  eps2=params[4]
  gf=2^dt
  raw=Array{Float64}(undef,N,round(Int,ceil(tf/dt)+1))
  d1=Normal(0.0,eps1)
  d2=Normal(0.5,eps2)
  init=Uniform(0.5*b/(2-a),2*b/(2-a))
  starts=rand(init,N)
  fins=max(a*starts+b+rand(d1,N),starts)
  t0=rand(Uniform(0,1),N)
  raw[:,1]=starts+(fins-starts).*t0
  div_waits=log(2,fins./starts).*(1-t0)
  t=0
  for j in 2:size(raw)[2]
    t=t+dt
    for i in 1:N
      div_waits[i]=div_waits[i]-dt
      if(div_waits[i]<0)
        div=min(max(rand(d2),0),1)
        if unbiased
          dtr=sample(1:(N+1))
        else
          dtr=i
        end
        if dtr<(N+1)
          starts[dtr]=fins[i]*(1-div)
          fins[dtr]=max(a*starts[dtr]+b+rand(d1),starts[dtr])
          raw[dtr,j]=starts[dtr]*2^(-div_waits[i])
          raw[dtr,j-1]=raw[dtr,j]
          div_waits[dtr]=div_waits[i]+log(2,fins[dtr]/starts[dtr])
        end
        if dtr!=i
          starts[i]=fins[i]*div
          raw[i,j]=starts[i]*2^(-div_waits[i])
          fins[i]=max(a*starts[i]+b+rand(d1),starts[i])
          div_waits[i]=div_waits[i]+log(2,fins[i]/starts[i])
        end
      else
        raw[i,j]=raw[i,j-1]*gf
      end
    end
  end
  return(raw[:,size(raw)[2]])
  #return raw
end
function nlm_grow(params,N,tf,dt)
  a=params[1]
  b=params[2]
  eps1=params[3]
  eps2=params[4]
  N0=max(round(Int,ceil(N/2^(tf))),1)
  raw=Array{Float64}(undef,N,round(Int,(tf+1)/dt))
  k=1
  d1=Normal(0.0,eps1)
  d2=Normal(0.5,eps2)
  init=Uniform(0.5*b/(2-a),2*b/(2-a))
  starts=Array{Float64}(undef,N)
  starts[1:N0]=rand(init,N0)
  fins=Array{Float64}(undef,N)
  fins[1:N0]=max(a*starts[1:N0]+b+rand(d1,N0),starts[1:N0])
  t0=rand(Uniform(0,1),N0)
  raw[1:N0,1]=starts[1:N0]+(fins[1:N0]-starts[1:N0]).*t0
  div_waits=Array{Float64}(undef,N)
  div_waits[1:N0]=log(2,fins[1:N0]./starts[1:N0]).*(1-t0)
  t=0
  Nt=N0
  while Nt<N
    t+=dt
    k+=1
    for i in 1:Nt
      div_waits[i]=div_waits[i]-dt
      if(div_waits[i]<0)
        div=min(max(rand(d2),0),1)
        starts[i]=fins[i]*div
        fins[i]=max(a*starts[i]+b+rand(d1),starts[i])
        raw[i,k]=starts[i]*2^(-div_waits[i])
        if Nt!=N
          Nt=Nt+1
          starts[Nt]=fins[i]*(1-div)
          fins[Nt]=max(a*starts[Nt]+b+rand(d1),starts[Nt])
          raw[Nt,k]=starts[Nt]*2^(-div_waits[i])
          #raw[Nt,j-1]=raw[Nt,j]
          div_waits[Nt]=div_waits[i]+log(2,fins[Nt]/starts[Nt])
        end
        div_waits[i]=div_waits[i]+log(2,fins[i]/starts[i])
      else
        raw[i,k]=raw[i,k-1]*2^(dt)
      end
    end
  end
  println(t)
  return(raw[:,k])
end
