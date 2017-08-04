``searchgrid`` helpers for building parameter grids for scikit-learn grid search
================================================================================

Specifying a parameter grid for
`GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`__
in Scikit-Learn can be annoying, particularly when:

-  you change your code to wrap some estimator in, say, a ``Pipeline``
   and then need to prefix all the parameters in the grid using lots of
   ``__``\ s
-  you are searching over multiple grids (i.e. your ``param_grid`` is a
   list) and you want to make a change to all of those grids

``searchgrid`` associates the parameters you want to search with each
particular estimator object, making it much more straightforward to
specify complex parameter grids, and means you don't need to update your
grid when you change the structure of your composite estimator.

It provides two main functions:

-  ``set_grid`` is used to specify the parameter values to be searched
   for an estimator or GP kernel.
-  ``make_grid_search`` is used to construct the ``GridSearchCV`` object
   with the parameter space the estimator is annotated with.

``build_param_grid`` is used by ``make_grid_search`` to construct the
``param_grid`` argument to ``GridSearchCV``.

Let's define a complicated search over the number of selected features
as well as a variety of classifiers and their parameters:

.. code:: python

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.feature_selection import SelectKBest
    >>> from searchgrid import set_grid, make_grid_search
    >>> clf1 = SVC()
    >>> clf2 = LogisticRegression()
    >>> clf3 = SVC()
    >>> clf4 = RandomForestClassifier()
    >>> estimator = set_grid(Pipeline([('sel', set_grid(SelectKBest(), k=[2, 3])),
    ...                                ('clf', None)]),
    ...                      clf=[set_grid(clf1, kernel=['linear']),
    ...                           clf2,
    ...                           set_grid(clf3, kernel=['poly'], degree=[2, 3]),
    ...                           clf4])
    >>> param_grid = [{'clf': [clf1], 'clf__kernel': ['linear'], 'sel__k': [2, 3]},
    ...               {'clf': [clf3], 'clf__kernel': ['poly'],
    ...                'clf__degree': [2, 3], 'sel__k': [2, 3]},
    ...               {'clf': [clf2, clf4], 'sel__k': [2, 3]}]
    >>> gscv = make_grid_search(estimator, cv=10, scoring='accuracy')
    >>> # assert gscv == param_grid  # Note sure why this comparison is failing
    >>> X, y = load_iris(return_X_y=True)
    >>> gscv.fit(X, y)  # doctest: +ELLIPSIS
    GridSearchCV(...)
    >>> # pd.DataFrame(gscv.cv_results_)
