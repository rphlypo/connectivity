import covariance_learn as covl
from nose.tools import *

# test the scoring methods
X = np.random.normal(size=(10, 6))

for alpha in linspace(0, 1, 5):
    gl = covl.GraphLasso(alpha=alpha, score_norm="ell0")
    gl.fit(X)
    assert_true(gl.score(gl.auxiliary_prec_) == 0)

    ips = covl.IPS(support=gl.auxiliary_prec_ != 0, score_norm="ell0")
    ips.fit(X)
    assert_true(ips.score(ips.auxiliary_prec_) == 0)

    hgl = covl.HierarchicalGraphLasso(htree=[[0, 1], [2, 3, 4]],
                                      alpha=alpha, score_norm="ell0")
    hgl.fit(X)
    assert_true(hgl.score(hgl.auxiliary_prec_) == 0)
