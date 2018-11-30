import pytest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline as skl_make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.datasets import load_iris
from searchgrid import set_grid, build_param_grid, make_grid_search
from searchgrid import make_column_transformer, make_pipeline, make_union


@pytest.mark.parametrize(('estimator', 'param_grid'), [
    (set_grid(SVC(), C=[1, 2]),
     {'C': [1, 2]}),
    (set_grid(SVC(), C=[1, 2], gamma=[1, 2]),
     {'C': [1, 2], 'gamma': [1, 2]}),
    (skl_make_pipeline(set_grid(SVC(), C=[1, 2], gamma=[1, 2])),
     {'svc__C': [1, 2], 'svc__gamma': [1, 2]}),
])
def test_build_param_grid(estimator, param_grid):
    assert build_param_grid(estimator) == param_grid


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


def test_step_estimator_grid_not_shared():
    # Fix for issue #10
    lr = set_grid(LogisticRegression(), C=[1, 2, 3])
    svc = SVC()
    grid = build_param_grid(set_grid(Pipeline([('root', lr)]), root=[lr, svc]))

    assert len(grid) == 2

    assert lr in grid[0]['root']
    assert svc not in grid[0]['root']
    assert 'root__C' in grid[0]

    assert svc in grid[1]['root']
    assert lr not in grid[1]['root']
    assert 'root__C' not in grid[1]


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


def test_make_pipeline():
    t1 = SelectKBest()
    t2 = SelectKBest()
    t3 = SelectKBest()
    t4 = SelectKBest()
    t5 = SelectPercentile()
    t6 = SelectKBest()
    t7 = SelectKBest()
    t8 = SelectKBest()
    t9 = SelectPercentile()
    in_steps = [[t1, None],
                [t2, t3],
                [t4, t5],  # mixed
                t6,
                [None, t7],
                [t8, None, t9],  # mixed
                None]
    pipe = make_pipeline(*in_steps, memory='/path/to/nowhere')
    union = make_union(*in_steps)

    for est, est_steps in [(pipe, pipe.steps),
                           (union, union.transformer_list)]:
        names, steps = zip(*est_steps)
        assert names == ('selectkbest-1', 'selectkbest-2', 'alt-1',
                         'selectkbest-3', 'selectkbest-4', 'alt-2', 'nonetype')
        assert steps == (t1, t2, t4, t6, None, t8, None)

        assert len(est._param_grid) == 5
        assert est._param_grid[names[0]] == [t1, None]
        assert est._param_grid[names[1]] == [t2, t3]
        assert est._param_grid[names[2]] == [t4, t5]
        assert est._param_grid[names[4]] == [None, t7]
        assert est._param_grid[names[5]] == [t8, None, t9]

    assert type(pipe) is Pipeline
    assert type(union) is FeatureUnion
    assert pipe.memory == '/path/to/nowhere'


def test_make_column_transformer():
    t1 = (SelectKBest(), ['column1'])
    t2 = (SelectKBest(), ['column2a', 'column2b'])
    t3 = (SelectKBest(), ['column3'])
    t4 = (SelectKBest(), ['column4'])
    t5 = (SelectPercentile(), ['column5'])
    t6 = (SelectKBest(), ['column6a', 'column6b'])
    t7 = (SelectKBest(), ['column7'])
    t8 = (SelectKBest(), ['column8'])
    t9 = (SelectPercentile(), ['column9'])

    in_steps = [[t1, None],
                [t2, t3],
                [t4, t5],  # mixed
                t6,
                [None, t7],
                [t8, None, t9],  # mixed
                None]
    column_transformer = make_column_transformer(*in_steps)
    names, steps = zip(*column_transformer.transformers)

    assert names == ('selectkbest-1', 'selectkbest-2', 'alt-1',
                     'selectkbest-3', 'selectkbest-4', 'alt-2', 'nonetype')
    assert steps == (t1, t2, t4, t6, None, t8, None)

    assert len(column_transformer._param_grid) == 5
    assert column_transformer._param_grid[names[0]] == [t1, None]
    assert column_transformer._param_grid[names[1]] == [t2, t3]
    assert column_transformer._param_grid[names[2]] == [t4, t5]
    assert column_transformer._param_grid[names[4]] == [None, t7]
    assert column_transformer._param_grid[names[5]] == [t8, None, t9]
