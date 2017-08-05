``searchgrid`` documentation
============================

Helps building parameter grids for :ref:`scikit-learn grid search
<scikit-learn:grid_search>`.

|version| |licence| |py-versions|

|build| |docs| |coverage|

Specifying a parameter grid for
:class:`~sklearn.model_selection.GridSearchCV`
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

``searchgrid`` allows you to define (and change) the grid together with the
esimator, reducing effort and sometimes code.

It provides two main functions:

-  :func:`searchgrid.set_grid` is used to specify the parameter values to be
   searched for an estimator or GP kernel.
-  :func:`searchgrid.make_grid_search` is used to construct the
   ``GridSearchCV`` object using the parameter space the estimator is annotated
   with.

Motivating examples
...................

Let's look over some of the messy change cases. We'll get some imports out of
the way.::

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.decomposition import PCA
    >>> from searchgrid import set_grid, make_grid_search
    >>> from sklearn.model_selection import GridSearchCV

Wrapping an estimator in a pipeline.
    You had code which searched over parameters for a classifier.
    Now you want to search for that classifier in a Pipeline.
    With plain old scikit-learn, you have to change::

        >>> gs = GridSearchCV(LogisticRegression(), {'C': [.1, 1, 10]})

    to::

        >>> gs = GridSearchCV(Pipeline([('reduce', SelectKBest()),
        ...                             ('clf', LogisticRegression())]),
        ...                   {'clf__C': [.1, 1, 10]})

    With ``searchgrid`` we only have to wrap our classifier in a Pipeline, and
    do not have to change the parameter grid, adding the ``clf__`` prefix. From::

        >>> lr = set_grid(LogisticRegression(), C=[.1, 1, 10])
        >>> gs = make_grid_search(lr)

    to::

        >>> lr = set_grid(LogisticRegression(), C=[.1, 1, 10])
        >>> gs = make_grid_search(Pipeline([('reduce', SelectKBest()),
        ...                                 ('clf', lr)]))


You want to change the estimator being searched in a pipeline.
    With scikit-learn, you have to change::

        >>> pipe = Pipeline([('reduce', SelectKBest()),
        ...                  ('clf', LogisticRegression())])
        >>> gs = GridSearchCV(pipe,
        ...                   {'reduce__k': [5, 10, 20],
        ...                    'clf__C': [.1, 1, 10]})

    to::

        >>> pipe = Pipeline([('reduce', PCA()),
        ...                  ('clf', LogisticRegression())])
        >>> gs = GridSearchCV(pipe,
        ...                   {'reduce__n_components': [5, 10, 20],
        ...                    'clf__C': [.1, 1, 10]})

    With ``searchgrid`` it's easier because you change the estimator and the
    parameters in the same place::

        >>> reduce = set_grid(SelectKBest(), k=[5, 10, 20])
        >>> lr = set_grid(LogisticRegression(), C=[.1, 1, 10])
        >>> pipe = Pipeline([('reduce', reduce),
        ...                  ('clf', lr)])
        >>> gs = make_grid_search(pipe)

    becomes::

        >>> reduce = set_grid(PCA(), n_components=[5, 10, 20])
        >>> lr = set_grid(LogisticRegression(), C=[.1, 1, 10])
        >>> pipe = Pipeline([('reduce', reduce),
        ...                  ('clf', lr)])
        >>> gs = make_grid_search(pipe)

Searching over multiple grids.
    You want to take the code from the previous example, but instead search
    over feature selection and PCA reduction in the same search.

    Without ``searchgrid``::

        >>> pipe = Pipeline([('reduce', None),
        ...                  ('clf', LogisticRegression())])
        >>> gs = GridSearchCV(pipe, [{'reduce': [SelectKBest()],
        ...                           'reduce__k': [5, 10, 20],
        ...                           'clf__C': [.1, 1, 10]},
        ...                          {'reduce': [PCA()],
        ...                           'reduce__n_components': [5, 10, 20],
        ...                           'clf__C': [.1, 1, 10]}])

    With ``searchgrid``::

        >>> kbest = set_grid(SelectKBest(), k=[5, 10, 20])
        >>> pca = set_grid(PCA(), n_components=[5, 10, 20])
        >>> lr = set_grid(LogisticRegression(), C=[.1, 1, 10])
        >>> pipe = set_grid(Pipeline([('reduce', None),
        ...                           ('clf', lr)]),
        ...                 reduce=[kbest, pca])
        >>> gs = make_grid_search(pipe)


.. |py-versions| image:: https://img.shields.io/pypi/pyversions/Django.svg
    :alt: Python versions supported

.. |version| image:: https://badge.fury.io/py/searchgrid.svg
    :alt: Latest version on PyPi
    :target: https://badge.fury.io/py/searchgrid

.. |build| image:: https://travis-ci.org/jnothman/searchgrid.svg?branch=master
    :alt: Travis CI build status
    :scale: 100%
    :target: https://travis-ci.org/jnothman/searchgrid

.. |coverage| image:: https://coveralls.io/repos/github/jnothman/searchgrid/badge.svg
    :alt: Test coverage
    :target: https://coveralls.io/github/jnothman/searchgrid

.. |docs| image:: https://readthedocs.org/projects/searchgrid/badge/?version=latest
     :alt: Documentation Status
     :scale: 100%
     :target: https://searchgrid.readthedocs.io/en/latest/?badge=latest

.. |licence| image:: https://img.shields.io/badge/Licence-BSD-blue.svg
     :target: https://opensource.org/licenses/BSD-3-Clause
