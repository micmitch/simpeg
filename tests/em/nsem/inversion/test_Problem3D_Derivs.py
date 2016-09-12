# Test functions
from glob import glob
import numpy as np, sys, os, time, scipy, subprocess
import SimPEG as simpeg
import unittest
from SimPEG import NSEM
from SimPEG.Utils import meshTensor
from scipy.constants import mu_0

np.random.seed(1983)

TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True


# Test the Jvec derivative
def DerivJvecTest(inputSetup,comp='All',freq=False,expMap=True):
    (M, freqs, sig, sigBG, rx_loc) = inputSetup
    survey, problem = NSEM.Utils.testUtils.setupSimpegNSEM_ePrimSec(inputSetup,comp=comp,singleFreq=freq,expMap=expMap)
    print 'Using {0} solver for the problem'.format(problem.Solver)
    print 'Derivative test of Jvec for eForm primary/secondary for {:s} comp at {:s}\n'.format(comp,survey.freqs)
    # problem.mapping = simpeg.Maps.ExpMap(problem.mesh)
    # problem.sigmaPrimary = np.log(sigBG)
    x0 = np.log(sigBG)
    # cond = sig[0]
    # x0 = np.log(np.ones(problem.mesh.nC)*cond)
    # problem.sigmaPrimary = x0
    # if True:
    #     x0  = x0 + np.random.randn(problem.mesh.nC)*cond*1e-1
    survey = problem.survey
    def fun(x):
        return survey.dpred(x), lambda x: problem.Jvec(x0, x)
    return simpeg.Tests.checkDerivative(fun, x0, num=3, plotIt=False, eps=FLR)

def DerivProjfieldsTest(inputSetup,comp='All',freq=False):

    survey, problem = NSEM.Utils.testUtils.setupSimpegNSEM_ePrimSec(inputSetup,comp,freq)
    print 'Derivative test of data projection for eFormulation primary/secondary\n\n'
    # problem.mapping = simpeg.Maps.ExpMap(problem.mesh)
    # Initate things for the derivs Test
    src = survey.srcList[0]

    u0x = np.random.randn(survey.mesh.nE)+np.random.randn(survey.mesh.nE)*1j
    u0y = np.random.randn(survey.mesh.nE)+np.random.randn(survey.mesh.nE)*1j
    u0 = np.vstack((simpeg.mkvc(u0x,2),simpeg.mkvc(u0y,2)))
    f0 = problem.fieldsPair(survey.mesh,survey)
    # u0 = np.hstack((simpeg.mkvc(u0_px,2),simpeg.mkvc(u0_py,2)))
    f0[src,'e_pxSolution'] =  u0[:len(u0)/2]#u0x
    f0[src,'e_pySolution'] = u0[len(u0)/2::]#u0y

    def fun(u):
        f = problem.fieldsPair(survey.mesh,survey)
        f[src,'e_pxSolution'] = u[:len(u)/2]
        f[src,'e_pySolution'] = u[len(u)/2::]
        return rx.eval(src,survey.mesh,f), lambda t: rx.evalDeriv(src,survey.mesh,f0,simpeg.mkvc(t,2))

    return simpeg.Tests.checkDerivative(fun, u0, num=3, plotIt=False, eps=FLR)



class NSEM_DerivTests(unittest.TestCase):

    def setUp(self):
        pass

    # Do a derivative test of Jvec
    def test_derivJvec_zxxr(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxxr',.1))
    def test_derivJvec_zxxi(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxxi',.1))
    def test_derivJvec_zxyr(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxyr',.1))
    def test_derivJvec_zxyi(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxyi',.1))
    def test_derivJvec_zyxr(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyxr',.1))
    def test_derivJvec_zyxi(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyxi',.1))
    def test_derivJvec_zyyr(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyyr',.1))
    def test_derivJvec_zyyi(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyyi',.1))
    # def test_derivJvec_All(self):self.assertTrue(DerivJvecTest(NSEM.Utils.testUtils.random(1e-2),'Imp',.1))

if __name__ == '__main__':
    unittest.main()
