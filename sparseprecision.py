# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as pyplt
import scipy as sp
import sklearn.covariance

def sparse_prec_cholesky( bool_mask):
    """Sparse Precision matrix based on given sparsity mask

    Keyword arguments:
    sparse_mask -- a boolean array that indicates the non-zero entries in the
                    precision matrix

    Set up a model for a sparse precision matrix based on a Cholesky decomposition.
    The decomposition is based on P = L L.T, with L a lower triangular matrix
    The goal is to fill L, such that P is positive definite fullfilling the imposed
    sparsity pattern given in sparse_mask

    Minimal example:
    bool_mask = create_Sparse_Pattern( p=5, sparse_frac=0.5)
    precision_matrix = sparse_Prec_Cholesky( bool_mask)

    Created on Mon Sep  9 10:47:02 2013
    @author: rphlypo
    """

    p = bool_mask.shape[0] # number of features, commonly denoted p
    L = np.random.normal(size=(p,p))
    L = np.tril( L, k=-1) + np.identity(p)

    #G = Gershgorin_disks(L,matsqrt=True)
    #plot_Gershgorin_disks(G)

    # invert boolean mask if all diagonal values are True
    # This is true if you used create_Sparse_Pattern
    if np.all( np.diag( bool_mask)):
        bool_mask = np.logical_not( bool_mask)

    # Because of symmetry we only need to deal with half of the entries
    zero_indices = np.nonzero( np.tril( bool_mask))

    P = np.zeros((p,p))
    for j in np.arange(p):
        i = zero_indices[0][zero_indices[1]==j]
        L[i,j] = -P[i,j]/L[j,j]
        P = P + np.outer( L[:,j], L[:,j])

    return P

def create_sparse_pattern( p=3, sparse_frac=0.3):
    """Create a symmetric mask of dimension pxp with given sparsity level

    Keyword arguments:
    p           -- dimension of the masked matrix will be pxp
    sparse_frac -- this is the fraction of zeros in the p(p-1)/2 lower triangle

    The (p,p) masked matrix will have [ sparce_frac x p(p-1) ] zeroes
    """
    nb_of_entries = np.ceil( p*(p-1)/2.*sparse_frac)

    valid_indices = np.nonzero(np.tril(np.ones((p,p)),k=-1))
    permute_indices = np.random.permutation(np.arange(p*(p-1)/2))
    valid_permute_indices = permute_indices[0:nb_of_entries]

    index_array = np.ones((p,p), dtype=bool)
    i_indices = valid_indices[0][valid_permute_indices]
    j_indices = valid_indices[1][valid_permute_indices]
    mask_entries = (np.concatenate((i_indices, j_indices), axis=0),
                    np.concatenate((j_indices, i_indices), axis=0))

    index_array[mask_entries] = False

    return index_array

def Gershgorin_disks( G, matsqrt=False):
    """Gives the Gershgoring disks associated with the matrix G

    Keyword arguments:
    matsqrt -- if a matrix "square root" is known for a symmetric G, one may
            give the factor, i.e., G = L L^T
    """
    if matsqrt:
        G = np.dot(G,G.T)
    Gabs = np.sum(np.abs(G),axis=1)
    Gdiag = np.abs(np.diag(G))
    G_disks = {'intervals':(2*Gdiag-Gabs,Gabs), 'center':Gdiag, 'radius':Gabs-Gdiag}
    return G_disks

def plot_Gershgorin_disks( G_disks):
    """ Plotting the Gershgorin disks

    plotting based on the return argument from :function:'Gershgorin_disks'
    """
    centers = G_disks['center']
    radii = G_disks['radius']
    pyplt.Figure()
    for i in np.arange(np.size(centers)):
        print 'circle %i: center = (%f,%f), radius = %f' % (i,centers[i], 0, radii[i])
        circle = pyplt.Circle((centers[i],0),radius=radii[i],color='r')
        pyplt.gca().add_artist(circle)
    pyplt.xlim( (np.min(centers-radii), np.max(centers+radii)))
    pyplt.ylim( (np.min(-radii), np.max(radii)))
    pyplt.show()

from sklearn.base import BaseEstimator, ClassifierMixin
class IterativeProportionalScaling(BaseEstimator,ClassifierMixin):
    def __init__(self, support=None, n_iter=1000, eps=1e-16, verbose=1,
                       prec_init=None, Ledoit_Wolf = False,
                       assume_centered=False):
        self.support = support
        self.n_iter = n_iter
        self.eps = eps
        self.verbose = verbose
        self.prec_init = prec_init
        self.Ledoit_Wolf = Ledoit_Wolf
        self.assume_centered = assume_centered

    def set_params(self,**kwargs):
        for param, val in kwargs.items():
            self.setattr(param,val)

    def get_params(self,deep=True):
        return {"support"   : self.support,
                "n_iter"    : self.n_iter,
                "eps"       : self.eps,
                "verbose"   : self.verbose,
                "prec_init" : self.prec_init}

    def fit(self,X,y=None,**kwargs):
        """ fit a sparse group precision matrix to the data in X


        The method uses iterative proportional scaling to match a precision
        matrix with a given support to the empirical covariance matrix.

        Parameters
        ----------
        X -- numpy ndarray of shape (n_samples, n_features)
        y -- None
                unused parameter, merely serves for compatibility as per the API
                of scikit-learn
        """
        X = np.asarray(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        if self.Ledoit_Wolf:
            cov_estim = sklearn.covariance.LedoitWolf(assume_centered=\
                                                        self.assume_centered)
        else:
            cov_estim = sklearn.covariance.EmpiricalCovariance(assume_centered=\
                                                        self.assume_centered)
        cov = cov_estim.fit(X).covariance_.copy()
        prec = self._iterative_proportional_scaling(cov,**kwargs)
        self.precision_ = prec
        return self


    def score(self,X):
        """ get the score of X under the current model

        Parameters
        ----------
        X -- numpy ndarray of shape [n_samples, n_features]

        Returns
        -------
        score -- float
                mean log-likelihood per dimension and per sample, i.e., the
                total log_likelihood divided by n_samples*n_features
        """
        emp_cov = sklearn.covariance.empirical_covariance(X - self.location_)
        return sklearn.covariance.log_likelihood(emp_cov,self.precision_)


    def _iterative_proportional_scaling(self, cov, **kwargs):
        """find ML estimate of precision matrix given covariance and graph G(A)

        Given a (possibly sparse) logical array A, estimate the precision
        matrix using the iterative proportional scaling method [1].

        Keyword arguments:
        -----------------
        Cov -- numpy ndarray shape=(n_features, n_features)
                the sample covariance matrix that is used as the sufficient
                statistic and which must be fitted
        A -- numpy ndarray, dtype=bool, shape=(n_features, n_features)
                logical array A as a matrix representation A(G) of the graph G
                the diagonal elements correspond to vertices, and off-diagonal
                elements denote the presence (True) or absence (False) of the
                corresponding edge in the graph G (as a consequence A is
                symmetric).
        K_init -- numpy ndarray, shape=(n_features, n_features)
                initialisation of the precision matrix
        n_iter -- unsigned integer
                the maximum number of iterations allowed before exiting the
                function
        eps -- (small) float
                required relative precision of solution upon exiting the
                function
        verbose -- unsigned integer
                the higher the verbose level, the more command line feedback
                you will get

        Returns:
        -------
        K -- numpy , shape=(n_features, n_features)
                maximum likelihood solution (if it exists) of the precision matrix
                given the sample covariance Cov and the graph representation A(G)

        References:
        ----------
        [1] Steffen F. Lauritzen: "Graphical Models", Clarendon Press, Oxford, 2004.
        """
        import networkx as nx

        # n_features is often referred to short-handed as 'p'
        p = cov.shape[0]

        support = self.support
        # Default is a fully connected graph
        if support is None:
            support = np.ones((p,p),dtype=bool)
        if not all(np.diag(support)):
            raise Exception("Bad Argument: "\
                            "'support' is not a connectivity matrix")

        # First we need to extract the cliques of the graph, using A(G)
        # we need to do this only once
        # Loop over the nodes, and search for cliques including the current node
        # but not the previous nodes (this cliques have already been added!!)
        # this introduces a lexicographic order on the vertices
        graph = _create_graph_from_connectivity(support)
        clique_set = list(nx.find_cliques(graph))
        if self.verbose > 1:
            print "{n} maximal cliques found".format(n=len(clique_set))
#        if plots:
#            pyplt.figure()
#            nx.draw_graphviz(graph)

#        sparsity_level = 1. - float(np.count_nonzero(support) - p)/p/(p-1)

        complete_edge_set = np.arange(p)

        # K is the estimate of the precision matrix, Cov the covariance
        if self.prec_init is None:
            prec = sp.linalg.pinv(np.diag(np.diag(cov)))
        else:
            prec = self.prec_init.copy()

        Loss = []
        LL = sklearn.covariance.log_likelihood
        Loss.append( -LL(cov, prec))

        iter_cnt = 0
        while True:
            iter_cnt += 1
            try:
                # Iterate over all cliques of G and update marginals
                for clique in clique_set:
                    # get complement of the maximal clique in the graph
                    clique_c = np.setdiff1d(complete_edge_set, clique)
                    if clique_c.size:
                        cc = np.ix_(clique,clique)
                        aa = np.ix_(clique_c,clique_c)
                        ca = np.ix_(clique,clique_c)
                        KSchur = np.dot(np.dot(prec[ca],sp.linalg.pinv(prec[aa])),
                                    prec[ca].T)/2.
                        KSchur += KSchur.T
                        prec[cc] = sp.linalg.pinv(cov[cc]) + KSchur
                    else:           # graph is complete, return the inverse
                        prec = sp.linalg.pinv(cov)
                        return prec
                Loss.append(-LL(cov, prec))
                attained_n_iter = (iter_cnt == self.n_iter)
                rel_diff = (Loss[-2] - Loss[-1])/Loss[-2]
                precision_at_solution = np.abs(rel_diff)
                attained_precision = (precision_at_solution < self.eps)
                if attained_precision or attained_n_iter:
                    raise StopIteration
            except StopIteration:
                break
        if self.verbose > 0 and attained_n_iter:
            print "No convergence reached in {0} iterations, "\
                  "returning current result".format(self.n_iter)
        if self.verbose > 1:
            print "Convergence reached in {0} iterations".format(iter_cnt)
        n_pos_diff = sum(np.diff(np.asarray(Loss))>0)
        if n_pos_diff > 0:
            Loss_array = np.asarray(Loss)
            abs_incr = np.diff(Loss_array)
            rel_incr = abs_incr / Loss_array[:-1]
            max_rel_incr = np.max( rel_incr)
            str_out = "objective function is rising {n} times\n"\
                      "\tlargest relative increase: {relinc}\n"\
                      "\tlargest absolute increase: {absinc}\n"
            print str_out.format(n=n_pos_diff,
                                 relinc=max_rel_incr,
                                 absinc=np.max(abs_incr))
        return prec

def _create_graph_from_connectivity(A):
    """ finds the maximal cliques of G under the representation A(G)

    maximal cliques are fully connected subgraphs, represented with a vertex
    set 'set_', such that `np.all(A[set_,:][:,set_])` is True and the addition
    of whatever vertex to the set would make the former evaluate to False

    We here return the maximal cliques as an iterator H

    Uses the Package NetworkX imported as nx
    """
    import networkx as nx
    # only symmetric part of A is needed to define the undirected edges
    # the non-zero entries then correspond to edges
    zero_ix = np.nonzero(np.tril(A))
    nb_edges = len(zero_ix[0])
    edges = [(zero_ix[0][i], zero_ix[1][i]) for i in np.arange(nb_edges)]
    G = nx.Graph()
    G.add_edges_from(edges)
    # just checking
    return G

def KL_loss( Kest, Sigma):
    """ Kullback_Leibler loss function directed from estimate to true model

    Kest is the estimated precision matrix, Sigma the optimal covariance
    """
    p = np.size(Kest,0)
    KS = np.dot(Kest, Sigma)
    eigKS = np.abs(np.linalg.eigvals( KS))
    eigKS = eigKS[eigKS > 1e-12]
    loss = np.sum(np.log(eigKS)) + np.trace(sp.linalg.pinv(KS)) - p
    return loss/2

def gen_sample_cov_LW(p,n):
    """Generate a sample covariance matrix of dimension p based on n samples
    """
    X = np.random.normal( loc=0., scale=1., size=(p, n))
    S = np.dot(X,X.T)/n
    coloring_mx = np.random.normal( loc=0., scale=1., size=(p,p))
    S = np.dot(np.dot(coloring_mx,S),coloring_mx.T)

    # The Ledoit-Wolf estimator
    mn = np.trace(S)/p
    dn2 = np.linalg.norm(S,ord='fro')**2/p - mn**2
    tmp = np.array([np.linalg.norm(np.outer(X[:,k],X[:,k])-S,ord='fro')**2 for k in np.arange(n)])
    bn2 = np.sum(tmp/p)/n**2
    bn2 = min(bn2,dn2)
    an2 = dn2-bn2

    rho1 = bn2/dn2*mn
    rho2 = an2/dn2

    S = rho1*np.identity(p) + rho2*S
    print "LW scalings:\n\trho1 (Id) = %.3f\n\trho2  (S) = %.3f" % (rho1, rho2)

    return S
