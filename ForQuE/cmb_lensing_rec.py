from headers import *
###############################################################################
# get noise power spectrum for lensing displacement d
# from CMB lens reconstruction. Follows Hu & Okamoto 2002.
# These are the N_l^dd. The noise on convergence kappa is:
# N_l^kappakappa = l^2/4. * N_l^dd

class CMBLensRec(object):

   def __init__(self, CMB, save=False, nProc=1):
      self.CMB = CMB
      self.nProc = nProc

      # bounds for ell integrals
      self.lMin = self.CMB.lMin
      self.lMax = max(self.CMB.lMaxT, self.CMB.lMaxP)
      
      # values of ell to compute the reconstruction noise
#       self.L = np.genfromtxt("./ForQuE/input/Lc.txt") # center of the bins for l
      self.L=np.unique(np.int32(np.logspace(np.log10(self.lMin),np.log10(self.lMax),20)))
      self.Nl = len(self.L)
      
      # output file path
      self.directory = "./output/cmblensrec/"+str(self.CMB)
      self.path = self.directory+"/cmblensrecnoise.txt"
      # create folder if needed
      if not os.path.exists(self.directory):
         os.makedirs(self.directory)

      if save or (not os.path.exists(self.path)):
         self.SaveAll()
#       self.LoadAll()
      
   
   def SaveAll(self):
      tStart0 = time()
      
      data = np.zeros((self.Nl, 17))
      data[:,0] = np.copy(self.L)
      
      # diagonal covariances
      #print ("Computing A_TT")
      tStart = time()
      data[:,1] = np.array(list(map(self.A_TT, self.L)))
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      
      #print ("Computing A_TE")
      tStart = time()
      data[:,2] = np.array(list(map(self.A_TE, self.L)))
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      #print ("Computing A_TB")
      tStart = time()
      data[:,3] = np.array(list(map(self.A_TB, self.L)))
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      #print ("Computing A_EE")
      tStart = time()
      data[:,4] = np.array(list(map(self.A_EE, self.L)))
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      #print ("Computing A_EB")
      tStart = time()
      data[:,5] = np.array(list(map(self.A_EB, self.L)))
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      
      # load diagonal covariances
      self.N_TT = data[:,1]
      self.N_TE = data[:,2]
      self.N_TB = data[:,3]
      self.N_EE = data[:,4]
      self.N_EB = data[:,5]
   
      # non-diagonal covariances
      #print ("Computing N_TT_TE")
      tStart = time()
      data[:,6] = np.array(list(map(self.N_TT_TE, self.L)))
      data[:,6] *= self.N_TT * self.N_TE / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      #print ("Computing N_TT_TB")
      tStart = time()
      data[:,7] = np.array(list(map(self.N_TT_TB, self.L)))
      data[:,7] *= self.N_TT * self.N_TB / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      #print ("Computing N_TT_EE")
      tStart = time()
      data[:,8] = np.array(list(map(self.N_TT_EE, self.L)))
      data[:,8] *= self.N_TT * self.N_EE / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      #print ("Computing N_TT_EB")
      tStart = time()
      data[:,9] = np.array(list(map(self.N_TT_EB, self.L)))
      data[:,9] *= self.N_TT * self.N_EB / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      #print ("Computing N_TE_TB")
      tStart = time()
      data[:,10] = np.array(list(map(self.N_TE_TB, self.L)))
      data[:,10] *= self.N_TE * self.N_TB / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
      #print ("took", (tStop-tStart)/60., "min")
      #print ("Computing N_TE_EE")
      tStart = time()
      data[:,11] = np.array(list(map(self.N_TE_EE, self.L)))
      data[:,11] *= self.N_TE * self.N_EE / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
#       #print "took", (tStop-tStart)/60., "min"
#       #print "Computing N_TE_EB"
      tStart = time()
      data[:,12] = np.array(list(map(self.N_TE_EB, self.L)))
      data[:,12] *= self.N_TE * self.N_EB / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
#       #print "took", (tStop-tStart)/60., "min"
#       #print "Computing N_TB_EE"
      tStart = time()
      data[:,13] = np.array(list(map(self.N_TB_EE, self.L)))
      data[:,13] *= self.N_TB * self.N_EE / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
#       #print "took", (tStop-tStart)/60., "min"
#       #print "Computing N_TB_EB"
      tStart = time()
      data[:,14] = np.array(list(map(self.N_TB_EB, self.L)))
      data[:,14] *= self.N_TB * self.N_EB / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
#       #print "took", (tStop-tStart)/60., "min"
#       #print "Computing N_EE_EB"
      tStart = time()
      data[:,15] = np.array(list(map(self.N_EE_EB, self.L)))
      data[:,15] *= self.N_EE * self.N_EB / self.L**2
      #np.savetxt(self.path, data)
      tStop = time()
#       #print "took", (tStop-tStart)/60., "min"

      # load non-diagonal covariances
      self.N_TT_TE = data[:,6]
      self.N_TT_TB = data[:,7]
      self.N_TT_EE = data[:,8]
      self.N_TT_EB = data[:,9]
      self.N_TE_TB = data[:,10]
      self.N_TE_EE = data[:,11]
      self.N_TE_EB = data[:,12]
      self.N_TB_EE = data[:,13]
      self.N_TB_EB = data[:,14]
      self.N_EE_EB = data[:,15]
      
      # variance of mv estimator
#       #print "Computing N_mv"
      tStart = time()
      self.SaveNmv(data)
      tStop = time()
#       #print "took", (tStop-tStart)/60., "min"
      
#       #print "Total time:", (tStop-tStart0)/3600., "hours"
   


   # Noise for minimum variance displacement estimator,
   # from Hu & Okamoto 2002
   def SaveNmv(self, data):
      Nmv = np.zeros(self.Nl)
      N = np.zeros((5, 5))
      for il in range(self.Nl):
         #print self.CMB
         #print il
         # fill in only the non-zero components
         N[0,0] = self.N_TT[il]
         N[0,1] = N[1,0] = self.N_TT_TE[il]
         N[0,3] = N[3,0] = self.N_TT_EE[il]
         #
         N[1,1] = self.N_TE[il]
         N[1,3] = N[3,1] = self.N_TE_EE[il]
         #
         N[2,2] = self.N_TB[il]
         N[2,4] = N[4,2] = self.N_TB_EB[il]
         #
         N[3,3] = self.N_EE[il]
         #
         N[4,4] = self.N_EB[il]
         #
         try:
            Inv = np.linalg.inv(N)
            Nmv[il] = 1./np.sum(Inv)
         except:
            pass
      data[:,16] = Nmv
      self.N_mv = data[:,16]
      forN_mv = UnivariateSpline(self.L, self.N_mv,k=1,s=0)
      self.fN_d_mv = lambda l: forN_mv(l)*(l>=min(self.L))*(l<=max(self.L))
      self.fN_k_mv = lambda l: l**2/4. * self.fN_d_mv(l)
      #np.savetxt(self.path, data)
      return



   def LoadAll(self):
      data = np.genfromtxt(self.path)
      # ell values
      self.L = data[:,0]
      self.Nl = len(self.L)
      # noises for displacement d
      # diagonal covariances
      self.N_TT = data[:,1]
      self.N_TE = data[:,2]
      self.N_TB = data[:,3]
      self.N_EE = data[:,4]
      self.N_EB = data[:,5]
      # non-diagonal covariances
      self.N_TT_TE = data[:,6]
      self.N_TT_TB = data[:,7]
      self.N_TT_EE = data[:,8]
      self.N_TT_EB = data[:,9]
      self.N_TE_TB = data[:,10]
      self.N_TE_EE = data[:,11]
      self.N_TE_EB = data[:,12]
      self.N_TB_EE = data[:,13]
      self.N_TB_EB = data[:,14]
      self.N_EE_EB = data[:,15]
      # variance of mv estimator
      self.N_mv = data[:,16]
   
      # interpolate the reconstruction noise N_l^dd, N_l^kappakappa and N_l^phiphi
      forN_TT = UnivariateSpline(self.L, self.N_TT,k=1,s=0)
      self.fN_d_TT = lambda l: forN_TT(l)*(l>=min(self.L))*(l<=max(self.L))
      self.fN_k_TT = lambda l: l**2/4. * self.fN_d_TT(l)
      self.fN_phi_TT = lambda l: self.fN_d_TT(l) / l**2
      #
      forN_mv = UnivariateSpline(self.L, self.N_mv,k=1,s=0)
      self.fN_d_mv = lambda l: forN_mv(l)*(l>=min(self.L))*(l<=max(self.L))
      self.fN_k_mv = lambda l: l**2/4. * self.fN_d_mv(l)
      self.fN_phi_mv = lambda l: self.fN_d_mv(l) / l**2
   
   
   ###############################################################################
   # functions f_alpha from Hu & Okamoto 2002
   # phi is the angle between l1 and l2
   
   def f_TT(self, l1, l2, phi):
      result = self.CMB.funlensedTT(l1) * l1*(l1 + l2*np.cos(phi))
      result += self.CMB.funlensedTT(l2) * l2*(l2 + l1*np.cos(phi))
      return result

   def f_TE(self, l1, l2, phi):
      # typo in Hu & Okamoto 2002: cos(2phi) and not cos(phi)!!!
      result = self.CMB.funlensedTE(l1) * l1*(l1 + l2*np.cos(phi)) * np.cos(2.*phi)
      result += self.CMB.funlensedTE(l2) * l2*(l2 + l1*np.cos(phi))
      return result

   def f_TB(self, l1, l2, phi):
      result = self.CMB.funlensedTE(l1) * l1*(l1 + l2*np.cos(phi)) * np.sin(2.*phi)
      return result

   def f_EE(self, l1, l2, phi):
      result = self.CMB.funlensedEE(l1) * l1*(l1 + l2*np.cos(phi))
      result += self.CMB.funlensedEE(l2) * l2*(l2 + l1*np.cos(phi))
      result *= np.cos(2.*phi)
      return result

   def f_EB(self, l1, l2, phi):
      result = self.CMB.funlensedEE(l1) * l1*(l1 + l2*np.cos(phi))
      result -= self.CMB.funlensedBB(l2) * l2*(l2 + l1*np.cos(phi))
      result *= np.sin(2.*phi)
      return result

   def f_BB(self, l1, l2, phi):
      result = self.CMB.funlensedBB(l1) * l1*(l1 + l2*np.cos(phi))
      result += self.CMB.funlensedBB(l2) * l2*(l2 + l1*np.cos(phi))
      result *= np.cos(2.*phi)
      return result
   
   

   ###############################################################################
   ###############################################################################
   # functions F_alpha from Hu & Okamoto 2002


   def F_TT(self, l1, l2, phi):
      result = self.f_TT(l1, l2, phi)
      result /= self.CMB.ftotalTT(l1)
      result /= self.CMB.ftotalTT(l2)
      result /= 2.
      if not np.isfinite(result):
         result = 0.
      return result

   def F_EE(self, l1, l2, phi):
      result = self.f_EE(l1, l2, phi)
      result /= self.CMB.ftotalEE(l1)
      result /= self.CMB.ftotalEE(l2)
      result /= 2.
      if not np.isfinite(result):
         result = 0.
      return result

   def F_BB(self, l1, l2, phi):
      result = self.f_BB(l1, l2, phi)
      result /= self.CMB.ftotalBB(l1)
      result /= self.CMB.ftotalBB(l2)
      result /= 2.
      if not np.isfinite(result):
         result = 0.
      return result


   def F_TB(self, l1, l2, phi):
      result = self.f_TB(l1, l2, phi)
      result /= self.CMB.ftotalTT(l1)
      result /= self.CMB.ftotalBB(l2)
      if not np.isfinite(result):
         result = 0.
      return result

   def F_EB(self, l1, l2, phi):
      result = self.f_EB(l1, l2, phi)
      result /= self.CMB.ftotalEE(l1)
      result /= self.CMB.ftotalBB(l2)
      if not np.isfinite(result):
         result = 0.
      return result

   def F_TE(self, l1, l2, phi):
      numerator = self.CMB.ftotalEE(l1) * self.CMB.ftotalTT(l2) * self.f_TE(l1, l2, phi)
      numerator -= self.CMB.ftotalTE(l1) * self.CMB.ftotalTE(l2) * self.f_TE(l2, l1, -phi)
      denom = self.CMB.ftotalTT(l1)*self.CMB.ftotalTT(l2) * self.CMB.ftotalEE(l1)*self.CMB.ftotalEE(l2)
      denom -= ( self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2) )**2
      result = numerator / denom
      if not np.isfinite(result):
         result = 0.
      return result

   ###############################################################################
   # A_alpha from Hu & Okamoto 2002
   # very important to enforce that lMin < l1,l2 < lMax
   # compute the integrals in ln(l) and not l, for speed
   # reduce the theta integral to [0,pi], by symmetry

   # returns the angle phi between l1 and l2,
   # given the angle theta between l1 and L=l1+l2
   def phi(self, L, l1, theta):
      x = L*np.cos(theta) - l1
      y = -L*np.sin(theta)  # y = L*np.sin(theta)
      result = np.arctan2(y,x)   # = 2.*np.arctan(y/(x+sqrt(x**2+y**2)))
      return result

   # returns the modulus of l2=L-l1,
   # given the moduli of L and l1 and the angle theta
   def l2(self, L, l1, theta):
      result = L**2 + l1**2 - 2.*L*l1*np.cos(theta)
      result = np.sqrt(result)
      return result
   
   # theta_min for the l1 integral
   # so that l2 > lMin
   def thetaMin(self, L, lnl1):
      l1 = np.log(lnl1)
      if (abs(L-l1)<self.lMin):
         theta_min = np.arccos((L**2+l1**2-self.lMin**2) / (2.*L*l1))
      else:
         theta_min = 0.
      return theta_min
   
   # theta_max for the l1 integral
   # so that l2 < lMax
   def thetaMax(self, L, lnl1):
      l1 = np.log(lnl1)
      if (l1>self.lMax-L):
         theta_max = np.arccos((L**2+l1**2-self.lMax**2) / (2.*L*l1))
      else:
         theta_max = np.pi
      return theta_max
   


   def A_TT(self, L):
      """Noise power spectrum of d_TT,
      the lensing deflection estimator from TT.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TT(l1, l2, phi) * self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_TT.__func__, "integ"):
         self.A_TT.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxT)], [0., np.pi]])
         self.A_TT.integ(integrand, nitn=8, neval=1000)

      result = self.A_TT.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean

      if not np.isfinite(result):
         result = 0.
      return result
   


   def A_TE(self, L):
      """Noise power spectrum of d_TE,
      the lensing deflection estimator from TE.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TE(l1, l2, phi) * self.F_TE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
   
      # if first time, initialize integrator
      if not hasattr(self.A_TE.__func__, "integ"):
         self.A_TE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.A_TE.integ(integrand, nitn=8, neval=1000)
      result = self.A_TE.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result
   
   

   def A_TB(self, L):
      """Noise power spectrum of d_TB,
      the lensing deflection estimator from TB.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TB(l1, l2, phi) * self.F_TB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result

      # if first time, initialize integrator
      if not hasattr(self.A_TB.__func__, "integ"):
         self.A_TB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.A_TB.integ(integrand, nitn=8, neval=1000)
      result = self.A_TB.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result


   def A_EE(self, L):
      """Noise power spectrum of d_EE,
      the lensing deflection estimator from EE.
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_EE(l1, l2, phi) * self.F_EE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_EE.__func__, "integ"):
         self.A_EE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.A_EE.integ(integrand, nitn=8, neval=1000)
      result = self.A_EE.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result


   def A_EB(self, L):
      """Noise power spectrum of d_EB,
      the lensing deflection estimator from EB.
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_EB(l1, l2, phi) * self.F_EB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_EB.__func__, "integ"):
         self.A_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.A_EB.integ(integrand, nitn=8, neval=1000)
      result = self.A_EB.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result

   def A_BB(self, L):
      """Noise power spectrum of d_BB,
      the lensing deflection estimator from BB.
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_BB(l1, l2, phi) * self.F_BB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_BB.__func__, "integ"):
         self.A_BB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.A_BB.integ(integrand, nitn=8, neval=1000)
      result = self.A_BB.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result


   ###############################################################################
   # N_alphabeta from Hu & Okamoto,
   # covariance of d_alpha and d_beta,
   # lensing deflection estimators from alpha and beta
   # without the factor A_alpha*A_beta/L^2,
   # for speed reasons.

   def N_TT_TE(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_TE(l1, l2, phi)*self.CMB.ftotalTT(l1)*self.CMB.ftotalTE(l2)
         result += self.F_TE(l2, l1, -phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTT(l2)
         result *= self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result

      # if first time, initialize integrator
      if not hasattr(self.N_TT_TE.__func__, "integ"):
         self.N_TT_TE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TT_TE.integ(integrand, nitn=8, neval=1000)
      result = self.N_TT_TE.integ(integrand, nitn=1, neval=5000)
      return result.mean

   def N_TT_TB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_TB(l1, l2, phi)*self.CMB.ftotalTT(l1)*self.CMB.ftotalTB(l2)
         result += 0.   #self.F_TB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalTT(l2)
         result *= self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TT_TB.__func__, "integ"):
         self.N_TT_TB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TT_TB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TT_TB.integ(integrand, nitn=1, neval=5000)
      return result.mean
      

   def N_TT_EE(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_EE(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2)
         result += self.F_EE(l2, l1, -phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2)
         result *= self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TT_TE.__func__, "integ"):
         self.N_TT_TE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TT_TE.integ(integrand, nitn=8, neval=1000)
      result = self.N_TT_TE.integ(integrand, nitn=1, neval=5000)
      return result.mean
      

   def N_TT_EB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_EB(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTB(l2)
         result += 0.   #self.F_EB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalTE(l2)
         result *= self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TT_EB.__func__, "integ"):
         self.N_TT_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TT_EB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TT_EB.integ(integrand, nitn=1, neval=5000)
      return result.mean


   def N_TE_TB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_TB(l1, l2, phi)*self.CMB.ftotalTT(l1)*self.CMB.ftotalEB(l2)
         result += 0.   #self.F_TB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalTE(l2)
         result *= self.F_TE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TE_TB.__func__, "integ"):
         self.N_TE_TB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TE_TB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TE_TB.integ(integrand, nitn=1, neval=5000)
      return result.mean


   def N_TE_EE(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_EE(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEE(l2)
         result += self.F_EE(l2, l1, -phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEE(l2)
         result *= self.F_TE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TE_EE.__func__, "integ"):
         self.N_TE_EE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TE_EE.integ(integrand, nitn=8, neval=1000)
      result = self.N_TE_EE.integ(integrand, nitn=1, neval=5000)
      return result.mean
      

   def N_TE_EB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_EB(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEB(l2)
         result += 0.   #self.F_EB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalEE(l2)
         result *= self.F_TE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TE_EB.__func__, "integ"):
         self.N_TE_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TE_EB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TE_EB.integ(integrand, nitn=1, neval=5000)
      return result.mean

   def N_TB_EE(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_EE(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEB(l2)
         result += 0.   #self.F_EE(l2, l1, -phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEB(l2)
         result *= self.F_TB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TB_EE.__func__, "integ"):
         self.N_TB_EE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TB_EE.integ(integrand, nitn=8, neval=1000)
      result = self.N_TB_EE.integ(integrand, nitn=1, neval=5000)
      return result.mean


   def N_TB_EB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_EB(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalBB(l2)
         result += 0.   #self.F_EB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalEB(l2)
         result *= self.F_TB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TB_EB.__func__, "integ"):
         self.N_TB_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TB_EB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TB_EB.integ(integrand, nitn=1, neval=5000)
      return result.mean


   def N_EE_EB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_EB(l1, l2, phi)*self.CMB.ftotalEE(l1)*self.CMB.ftotalEB(l2)
         result += 0.   #self.F_EB(l2, l1, -phi)*self.CMB.ftotalEB(l1)*self.CMB.ftotalEE(l2)
         result *= self.F_EE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_EE_EB.__func__, "integ"):
         self.N_EE_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.N_EE_EB.integ(integrand, nitn=8, neval=1000)
      result = self.N_EE_EB.integ(integrand, nitn=1, neval=5000)
      return result.mean


   ###############################################################################
   # plots


   def plotNoise(self, fPkappa=None):
      # diagonal covariances: all
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      if fPkappa is None:
         # read C_l^phiphi for the Planck cosmology
         data = np.genfromtxt("./ForQuE/input/universe_Planck15/camb/lenspotentialCls.dat")
         L = data[:,0]
         Pphi = data[:, 5] * (2.*np.pi) / L**4
         ax.plot(L, L**4/(2.*np.pi)*Pphi, 'k-', lw=3, label=r'signal')
      else:
         Pkappa = np.array(list(map(fPkappa, self.L)))
         ax.plot(self.L, Pkappa * 4./(2.*np.pi), 'k-', lw=3, label=r'signal')
      #
      ax.plot(self.L, self.L**2 * self.N_TT/(2.*np.pi), c=plt.cm.rainbow(0.), lw=1.5, label=r'TT')
      ax.plot(self.L, self.L**2 * self.N_TE/(2.*np.pi), c=plt.cm.rainbow(1./5.), lw=1.5, label=r'TE')
      ax.plot(self.L, self.L**2 * self.N_TB/(2.*np.pi), c=plt.cm.rainbow(2./5.), lw=1.5, label=r'TB')
      ax.plot(self.L, self.L**2 * self.N_EE/(2.*np.pi), c=plt.cm.rainbow(3./5.), lw=1.5, label=r'EE')
      ax.plot(self.L, self.L**2 * self.N_EB/(2.*np.pi), c=plt.cm.rainbow(4./5.), lw=1.5, label=r'EB')
      ax.plot(self.L, self.L**2 * self.N_mv/(2.*np.pi), 'r', lw=3, label=r'min. var.')
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='mask')
      ax.set_xlabel(r'$L$', fontsize=24)
      ax.set_ylabel(r'$L^2C^{dd}_L / (2\pi) = 4C_L^\kappa  / (2\pi)$', fontsize=24)



      # diagonal covariances: mv versus signal
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.loglog(L, L**4/(2.*np.pi)*Pphi, 'k', lw=3, label=r'signal')
      #
      ax.loglog(self.L, self.L**2 * self.N_mv/(2.*np.pi), 'r', lw=3, label=r'mv')
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell^2 C^{dd}_\ell / (2\pi)$')
      #ax.set_ylim((1.e-9, 1.e-5))
      ax.set_xlim((self.lMin, self.lMax))
      ax.set_title(r'Noise in lensing deflection reconstruction')
      #
      path = "./figures/cmblensrec/"+str(self.CMB)+"/short_recnoise_lmax"+str(int(self.lMax))+".pdf"
      #fig.savefig(path, bbox_inches='tight')
      

      # non-diagonal covariances
      fig=plt.figure(2)
      ax=fig.add_subplot(111)
      #
      ax.loglog(self.L, self.L**2 * np.abs(self.N_TT_TE)/(2.*np.pi), 'k', lw=2, label=r'TT-TE')
      ax.loglog(self.L, self.L**2 * np.abs(self.N_TT_EE)/(2.*np.pi), 'r', lw=2, label=r'TT-EE')
      ax.loglog(self.L, self.L**2 * np.abs(self.N_TE_EE)/(2.*np.pi), 'g', lw=2, label=r'TE-EE')
      ax.loglog(self.L, self.L**2 * np.abs(self.N_TB_EB)/(2.*np.pi), 'c', lw=2, label=r'TB-EB')
      #
      ax.legend(loc=2)
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell^2 | C_\ell | / (2\pi)$')
      ax.set_title(r'Noise in lensing deflection reconstruction')
      #
      path = "./figures/cmblensrec/"+str(self.CMB)+"/cross_recnoise_lmax"+str(int(self.lMax))+".pdf"
      #fig.savefig(path, bbox_inches='tight')
      
      plt.show()




