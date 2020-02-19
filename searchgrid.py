from collections import Mapping as _Mapping
from collections import defaultdict as _defaultdict
import itertools as _itertools

from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.model_selection import GridSearchCV as _GridSearchCV
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.pipeline import FeatureUnion as _FeatureUnion


def set_grid(estimator, **grid):
    """Set the grid to search for the specified estimator

    Overwrites any previously set grid.

    Parameters
    ----------
    grid : dict (str -> list of values)
        Keyword arguments define the values to be searched for each specified
        parameter.

    Returns
    -------
    estimator
        Useful for chaining
    """
    estimator._param_grid = grid
    return estimator


def _update_grid(dest, src, prefix=None):
    # TODO: needs docs
    if src is None:
        return dest
    if prefix:
        src = [{prefix + k: v for k, v in d.items()}
               for d in src]
    out = []
    for d1, d2 in _itertools.product(dest, src):
        out_d = d1.copy()
        out_d.update(d2)
        out.append(out_d)
    return out


def _build_param_grid(estimator):
    grid = getattr(estimator, '_param_grid', {})
    if isinstance(grid, _Mapping):
        grid = [grid]

    # handle estimator parameters having their own grids
    for param_name, value in estimator.get_params().items():
        if '__' not in param_name and hasattr(value, 'get_params'):
            out = []
            value_grid = _build_param_grid(value)
            for sub_grid in grid:
                if param_name in sub_grid:
                    sub_grid = [sub_grid]
                else:
                    sub_grid = _update_grid([sub_grid], value_grid,
                                            param_name + '__')
                out.extend(sub_grid)
            grid = out

    # handle grid values having their own grids
    out = []
    for out_d in grid:
        part = [out_d]
        for param_name, values in out_d.items():
            to_update = []
            no_sub_grid = []
            for v in values:
                if hasattr(v, 'get_params'):
                    sub_grid = _build_param_grid(v)
                    if sub_grid is not None:
                        to_update.extend(_update_grid([{param_name: [v]}],
                                                      sub_grid,
                                                      param_name + '__'))
                        continue
                no_sub_grid.append(v)

            if no_sub_grid:
                to_update.append({param_name: no_sub_grid})

            part = _update_grid(part, to_update)
        out.extend(part)

    if out == [{}]:
        return None

    return out


def build_param_grid(estimator):
    """Determine the parameter grid annotated on the estimator

    Parameters
    ----------
    estimator : scikit-learn compatible estimator
        Should have been annotated using :func:`set_grid`

    Notes
    -----
    Most often, it is unnecessary for this to be used directly, and
    :func:`make_grid_search` should be used instead.
    """
    out = _build_param_grid(estimator)
    if out is None:
        return {}
    elif len(out) == 1:
        return out[0]
    return out


def _check_estimator(estimator):
    if isinstance(estimator, list):
        estimator = set_grid(_Pipeline([('root', estimator[0])]),
                             root=estimator)
    elif not hasattr(estimator, 'fit'):
        raise ValueError('Expected estimator, but %r does not have .fit'
                         % estimator)
    return estimator


def make_grid_search(estimator, **kwargs):
    """Construct a GridSearchCV with the given estimator and its set grid

    Parameters
    ----------
    estimator : (list of) estimator
        When a list, the estimators are searched over.
    kwargs
        Other parameters to the
        :class:`sklearn.model_selection.GridSearchCV` constructor.
    """
    estimator = _check_estimator(estimator)
    return _GridSearchCV(estimator, build_param_grid(estimator), **kwargs)


def _name_steps(steps, default='alt'):
    """Generate names for estimators."""
    steps = [estimators if isinstance(estimators, list) else [estimators]
             for estimators in steps]

    names = []
    for estimators in steps:
        estimators = estimators[:]
        if len(estimators) > 1:
            while None in estimators:
                estimators.remove(None)
        step_names = {_name_of_estimator(estimator)
                      for estimator in estimators}
        if len(step_names) > 1:
            names.append(default)
        else:
            names.append(step_names.pop())

    namecount = _defaultdict(int)
    for name in names:
        namecount[name] += 1

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(names))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    named_steps = list(zip(names, [step[0] for step in steps]))
    grid = {k: v for k, v in zip(names, steps) if len(v) > 1}
    return named_steps, grid


def _name_of_estimator(estimator):
    if isinstance(estimator, tuple):
        # tuples comes from ColumnTransformers. At the moment, sklearn accepts
        # both (estimator, list_of_columns) and (list_of_columns, estimator)
        tuple_types = {type(tuple_entry) for tuple_entry in estimator}
        tuple_types.discard(list)
        estimator_type = tuple_types.pop()
    else:
        estimator_type = type(estimator)

    return estimator_type.__name__.lower()


def make_pipeline(*steps, **kwargs):
    """Construct a Pipeline with alternative estimators to search over

    Parameters
    ----------
    steps
        Each step is specified as one of:

        * an estimator instance
        * None (meaning no transformation)
        * a list of the above, indicating that a grid search should alternate
          over the estimators (or None) in the list
    kwargs
        Keyword arguments to the constructor of
        :class:`sklearn.pipeline.Pipeline`.

    Examples
    --------
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.feature_extraction.text import TfidfTransformer
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import ParameterGrid
    >>> from searchgrid import make_pipeline, build_param_grid
    >>> pipe = make_pipeline(CountVectorizer(),
    ...                      [TfidfTransformer(), None],
    ...                      [PCA(n_components=5), SelectKBest(k=5)],
    ...                      [set_grid(LogisticRegression(),
    ...                                C=[.1, 1., 10.]),
    ...                       RandomForestClassifier()])
    >>> pipe.steps  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [('countvectorizer', CountVectorizer(...)),
     ('tfidftransformer', TfidfTransformer(...)),
     ('alt-1', PCA(...)),
     ('alt-2', LogisticRegression(...))]
    >>> n_combinations = len(ParameterGrid(build_param_grid(pipe)))
    >>> n_combinations
    ... # 2 * 2 * (3 + 1)
    16

    Notes
    -----
    Each step is named according to the set of estimator types in its list:

    * if a step has only one type of estimator (disregarding None), it takes
      that estimator's class name (lowercased)
    * if a step has estimators of mixed type, the step is named 'alt'
    * if there are multiple steps of the same name using the above rules,
      a suffix '-1', '-2', etc. is added.
    """
    steps, grid = _name_steps(steps)
    return set_grid(_Pipeline(steps, **kwargs), **grid)


def make_union(*transformers, **kwargs):
    """Construct a FeatureUnion with alternative estimators to search over

    Parameters
    ----------
    steps
        Each step is specified as one of:

        * an estimator instance
        * None (meaning no features)
        * a list of the above, indicating that a grid search should alternate
          over the estimators (or None) in the list
    kwargs
        Keyword arguments to the constructor of
        :class:`sklearn.pipeline.FeatureUnion`.

    Notes
    -----
    Each step is named according to the set of estimator types in its list:

    * if a step has only one type of estimator (disregarding None), it takes
      that estimator's class name (lowercased)
    * if a step has estimators of mixed type, the step is named 'alt'
    * if there are multiple steps of the same name using the above rules,
      a suffix '-1', '-2', etc. is added.
    """
    steps, grid = _name_steps(transformers)
    return set_grid(_FeatureUnion(steps, **kwargs), **grid)


def make_column_transformer(*transformers, **kwargs):
    """Construct a ColumnTransformer with alternative estimators to search over

    Parameters
    ----------
    steps
        Each step is specified as one of:

        * an (estimator, [column_names]) or ([column_names], estimator) tuple
        * None (meaning no features)
        * a list of the above, indicating that a grid search should alternate
          over the estimators (or None) in the list
    kwargs
        Keyword arguments to the constructor of
        :class:`sklearn.pipeline.FeatureUnion`.

    Notes
    -----
    Each step is named according to the set of estimator types in its list:

    * if a step has only one type of estimator (disregarding None), it takes
      that estimator's class name (lowercased)
    * if a step has estimators of mixed type, the step is named 'alt'
    * if there are multiple steps of the same name using the above rules,
      a suffix '-1', '-2', etc. is added.
    """
    steps, grid = _name_steps(transformers)
    return set_grid(_ColumnTransformer(steps, **kwargs), **grid)
