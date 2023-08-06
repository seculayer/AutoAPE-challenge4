require(data.table)
require(bit64)
require(dbscan)
require(doParallel)
require(rBayesianOptimization)
path='../input/train_1/'

score<-function(sub,dft){
  df=merge(sub,dft[,.(hit_id,particle_id,weight)])
  df[,Np:=.N,by=particle_id]# Np = Hits per Particle
  df[,Nt:=.N,by=track_id]   # Nt = Hits per Track
  df[,Ntp:=.N,by=list(track_id,particle_id)]# Hits per Particle per Track
  df[,r1:=Ntp/Nt]
  df[,r2:=Ntp/Np]
  sum(df[r1>.5 & r2>.5,weight])
}

trackML <- function(dfh,w1,w2,w3,Niter){
  dfh[,s1:=hit_id]
  dfh[,N1:=1L] 
  dfh[,r:=sqrt(x*x+y*y+z*z)]
  dfh[,rt:=sqrt(x*x+y*y)]
  dfh[,a0:=atan2(y,x)]
  dfh[,z1:=z/rt]
  dfh[,z2:=z/r]
  mm     <-  1
  for (ii in 0:Niter) {
    mm <- mm*(-1)
    dfh[,a1:=a0+mm*(rt+0.000005*rt^2)/1000*(ii/2)/180*pi]
    dfh[,sina1:=sin(a1)]
    dfh[,cosa1:=cos(a1)]
    dfs=scale(dfh[,.(sina1,cosa1,z1,z2)])
	cx <- c(w1,w1,w2,w3)
    for (jj in 1:ncol(dfs)) dfs[,jj] <- dfs[,jj]*cx[jj]
    res=dbscan(dfs,eps=0.0035,minPts = 1)
    dfh[,s2:=res$cluster]
    dfh[,N2:=.N, by=s2]
    maxs1 <- max(dfh$s1)
    dfh[,s1:=ifelse(N2>N1 & N2<20,s2+maxs1,s1)]
    dfh[,s1:=as.integer(as.factor(s1))]
    dfh[,N1:=.N, by=s1]    
  }
  return(dfh$s1)
}
#######################################
# function for Bayessian Optimization #
#   (needs lists: Score and Pred)     #
#######################################
Fun4BO <- function(w1,w2,w3,Niter) { 
   dfh$s1 <- trackML(dfh,w1,w2,w3,Niter)
   sub=data.table(event_id=nev,hit_id=dfh$hit_id,track_id=dfh$s1)
   sc <- score(sub,dft)
   list(Score=sc,Pred=0)
}

print("Bayesian Optimization")
nev=1000
dfh=fread(paste0(path,'event00000',nev,'-hits.csv'))
dft=fread(paste0(path,'event00000',nev,'-truth.csv'),stringsAsFactors = T)
OPT <- BayesianOptimization(Fun4BO,
   bounds = list(w1 = c(0.9, 1.2), w2 = c(0.3, 0.7), w3 = c(0.1, 0.4), Niter = c(140L, 190L)),
   init_points = 3, n_iter = 20,
   acq = "ucb", kappa = 2.576, eps = 0.0,
   verbose = TRUE)


namef <- system("cd ../input/test; ls *hits.csv", intern=TRUE)
path <- '../input/test/'
print("Preparing submission")
w1    <- OPT$Best_Par[[1]]
w2    <- OPT$Best_Par[[2]]
w3    <- OPT$Best_Par[[3]]
Niter <- OPT$Best_Par[[4]]
registerDoParallel(cores=4)
print("Parallel") # wait until "Finished"
sub <- foreach(nev = 0:124, .combine = rbind)  %dopar%  {
         dfh <- fread(paste0(path,namef[nev+1]))
         dfh$s1 <- trackML(dfh,w1,w2,w3,Niter)
         subEvent <- data.table(event_id=nev,hit_id=dfh$hit_id,track_id=dfh$s1)
         return(subEvent)    
       }
fwrite(sub, "submission.csv")
print('Finished')

