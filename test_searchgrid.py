import pytest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import load_iris
from searchgrid import set_grid, build_param_grid, make_grid_search


@pytest.mark.parametrize(('estimator', 'param_grid'), [
    (set_grid(SVC(), C=[1, 2]),
     {'C': [1, 2]}),
    (set_grid(SVC(), C=[1, 2], gamma=[1, 2]),
     {'C': [1, 2], 'gamma': [1, 2]}),
    (make_pipeline(set_grid(SVC(), C=[1, 2], gamma=[1, 2])),
     {'svc__C': [1, 2], 'svc__gamma': [1, 2]}),
])
def test_build_param_grid(estimator, param_grid):
    assert build_param_grid(estimator) == param_grid

#    pytest.mark.xfail(
#        (set_grid(SVC(), [{'kernel': ['linear']},
#                          {'kernel': 'rbf', 'gamma': [1, 2]}]),
#         [{'kernel': ['linear']}, {'kernel': 'rbf', 'gamma': [1, 2]}])),
#    pytest.mark.xfail(
#        (make_pipeline(set_grid(SVC(), [{'kernel': ['linear']},
#                                        {'kernel': ['rbf'],
#                                         'gamma': [1, 2]}])),
#         [{'svc__kernel': ['linear']},
#          {'svc__kernel': 'rbf', 'svc__gamma': [1, 2]}])),


def test_build_param_grid_set_estimator():
    clf1 = SVC()
    clf2 = LogisticRegression()
    clf3 = SVC()
    clf4 = SGDClassifier()
    estimator = set_grid(Pipeline([('sel', set_grid(SelectKBest(), k=[2, 3])),
                                   ('clf', None)]),
                         clf=[set_grid(clf1, kernel=['linear']),
                              clf2,
                              set_grid(clf3, kernel=['poly'], degree=[2, 3]),
                              clf4])
    param_grid = [{'clf': [clf1], 'clf__kernel': ['linear'], 'sel__k': [2, 3]},
                  {'clf': [clf3], 'clf__kernel': ['poly'],
                   'clf__degree': [2, 3], 'sel__k': [2, 3]},
                  {'clf': [clf2, clf4], 'sel__k': [2, 3]}]
    assert build_param_grid(estimator) == param_grid


def test_make_grid_search():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression()
    svc = set_grid(SVC(), kernel=['poly'], degree=[2, 3])
    gs1 = make_grid_search(lr, cv=5)  # empty grid
    gs2 = make_grid_search(svc, cv=5)
    gs3 = make_grid_search([lr, svc], cv=5)
    for gs, n_results in [(gs1, 1), (gs2, 2), (gs3, 3)]:
        gs.fit(X, y)
        assert gs.cv == 5
        assert len(gs.cv_results_['params']) == n_results

    svc_mask = gs3.cv_results_['param_root'] == svc
    assert svc_mask.sum() == 2
    assert gs3.cv_results_['param_root__degree'][svc_mask].tolist() == [2, 3]
    assert gs3.cv_results_['param_root'][~svc_mask].tolist() == [lr]
