# -*- coding: utf-8 -*-
"""
@author: Guilherme Mazanti

We implement here some methods for testing stability and controllability
properties of difference equations, based on the articles:

[1] Y. Chitour, G. Mazanti, M. Sigalotti.
Stability of non-autonomous difference equations with applications to transport
and wave propagation on networks.
Netw. Heterog. Media, 11(4):563–601, 2016.

[2] G. Mazanti.
Relative controllability of linear difference equations.
SIAM J. Control Optim., 55(5):3132–3153, 2017.

[3] Y. Chitour, G. Mazanti, M. Sigalotti.
Approximate and exact controllability of linear difference equations.
J. Éc. polytech. Math., 7:93–142, 2020.

TODO:
  - Take into account rational dependece structure of the delays
  - Implement approximate and exact controllability tests in dimension 2 with 
    2 delays
  - Implement stability tests (Hale--Silkowski to begin with)
  - Implement time-dependent stability tests
  - Implement results for generic systems under a given structure
"""

import numpy as np
from scipy.special import binom

class MaxSumIterator:
  """
  Iterates over all tuples n == (n[0], n[1], ..., n[N-1]) such that
  sum(n) <= maxSum
  """
  def __init__(self, N, maxSum):
    self.current = np.zeros(N, dtype=int)
    self.current[-1] = -1
    self.maxSum = maxSum
    
  def __iter__(self):
    return self
  
  def __len__(self):
    length = 0
    for k in range(self.maxSum+1):
      length += int(binom(self.current.size-1+k, k))
    return length
  
  def __next__(self):      
    if self.current.sum() < self.maxSum:
      self.current[-1] += 1
      return tuple(int(x) for x in self.current)
    
    if self.current[0]==self.maxSum:
      raise StopIteration
    
    # Find the first index from the end which is non-zero
    i = self.current.nonzero()[0][-1]
    self.current[i-1] += 1
    self.current[i] = 0
    return tuple(int(x) for x in self.current)

class DifferenceEquation:
  def __init__(self, A, B, Lambda, copy = True):
    """
    A: a 3d numpy array of shape (d, d, N) such that A[:, :, i] contains the
       system matrix Ai
    B: a 2d numpy array of shape (d, m) containing the control matrix
    Lambda: a 1d numpy array of shape (N,) with the delays
    copy: if True, the class will store copies of A, B, and Lambda. Otherwise,
    only references to A, B, and Lambda are stored.
    """
    assert type(A)==np.ndarray and type(B)==np.ndarray and\
           type(Lambda)==np.ndarray, "A, B, and Lambda must be numpy arrays."
    assert A.ndim==3, "A must be a 3d numpy array"
    assert B.ndim==2, "B must be a 2d numpy array"
    assert Lambda.ndim==1, "Lambda must be a 1d numpy array"
    self.d = A.shape[0]
    self.N = A.shape[2]
    self.m = B.shape[1]
    assert A.shape[1]==self.d, "A must be of shape (d, d, N)"
    assert B.shape[0]==self.d, "A and B must have the same first dimension"
    assert Lambda.shape[0]==self.N, "The size of Lambda must be equal to the"+\
           " 3rd dimension of A"
           
    if copy:
      self.A = A.copy()
      self.B = B.copy()
      self.Lambda = Lambda.copy()
    else:
      self.A = A
      self.B = B
      self.Lambda = Lambda
      
    self.Xi = {tuple([0 for _ in range(self.N)]): np.eye(self.d)}
  
  def _requiredIndices(self, n):
    """
    Given an index n, returns all indices required to compute Xi_n by the
    inductive formuma for Xi_n (see [2, (2.2)]). These indices are returned by
    a dictionnary indexed by the index of the matrix A_k multiplying the
    corresponding coefficient in [2, (2.2)].
    
    This function does not verify if its argument n has the required structure.
    
    Input:
      n: tuple with N nonnegative integers.
      
    Output:
      Dictionnary req such that, for every k in range(N), req[k] contains
      n - ek if and only if all entries of the latter are nonnegative. Here,
      e0, ..., eN-1 is the canonical basis of R^N.
    """
    req = dict()
    for k in range(self.N):
      if n[k] >= 1:
        ek = np.zeros(self.N, dtype=int)
        ek[k] = 1
        req[k] = tuple(np.array(n) - ek)
    return req
      
  
  def getXi(self, n):
    """
    Returns the coefficient Xi_n
    
    Input:
      n: tuple with N nonnegative integers
      
    Output:
      Coefficient Xi_n, as a 2d (d, d) numpy array.
    """
    # If the coefficient was already computed and stored in self.Xi, we simply
    # return it. Otherwise, we use the recursion formula. We avoid creating a
    # recursive function by explicitly creating a stack
    assert type(n)==tuple and len(n)==self.N, "n must be a tuple with N"+\
           " elements"
    for ni in n:
      assert type(ni)==int and ni >= 0, "The entries of n must be nonnegative"+\
             " integers"
             
    if n not in self.Xi:
      stack = [n]
      while stack != []:
        m = stack.pop()
        
        # 1st step: check if all required coefficients have already been
        # computed
        req = self._requiredIndices(m)
        allComputed = True
        for k, mMinusEk in req.items():
          if mMinusEk not in self.Xi:
            if allComputed:
              allComputed = False
              stack.append(m)
            stack.append(mMinusEk)
        
        if allComputed:
          self.Xi[m] = np.zeros((self.d, self.d))
          for k, mMinusEk in req.items():
            self.Xi[m] += self.A[:, :, k].dot(self.Xi[mMinusEk])
    return self.Xi[n]
  
  def isRelativelyControllable(self):
    """
    Tests if the system is relatively controllable with rationally independent
    delays using [2, Corollary 5.8].
    """
    nIter = MaxSumIterator(self.N, self.d-1)
    M = np.empty((self.d, self.m*len(nIter)))
    col = 0
    for n in nIter:
      M[:, col:(col+self.m)] = self.getXi(n).dot(B)
      col += self.m
    return np.linalg.matrix_rank(M)==self.d
    
# =============================================================================
# Tests
# =============================================================================
if __name__=="__main__":
  A = np.empty((3, 3, 2))
  A[:, :, 0] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
  A[:, :, 1] = np.array([[-2, 0, 3], [0, 1, 0], [-5, 0, 8]])
  B = np.array([[0], [0], [1]])
  Lambda = np.random.rand(2)*10
  System = DifferenceEquation(A, B, Lambda)
  print(System.isRelativelyControllable())