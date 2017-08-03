from collections import Mapping
import itertools

from sklearn.model_selection import GridSearchCV as _GridSearchCV


def set_grid(estimator, **grid):
    estimator._param_grid = grid
    return estimator


def _update_grid(dest, src, prefix=None):
    if src is None:
        return dest
    if prefix:
        src = [{prefix + k: v for k, v in d.items()}
               for d in src]
    out = []
    for d1, d2 in itertools.product(dest, src):
        out_d = d1.copy()
        out_d.update(d2)
        out.append(out_d)
    return out


def _build_param_grid(estimator):
    grid = getattr(estimator, '_param_grid', {})
    if isinstance(grid, Mapping):
        grid = [grid]

    # handle estimator parameters having their own grids
    for param_name, value in estimator.get_params().items():
        if '__' not in param_name and hasattr(value, 'get_params'):
            grid = _update_grid(grid, _build_param_grid(value),
                                param_name + '__')

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
    print(out)

    if out == [{}]:
        return None

    return out


def build_param_grid(estimator):
    out = _build_param_grid(estimator)
    if out is None:
        return {}
    elif len(out) == 1:
        return out[0]
    return out


def make_grid_search(estimator, **kwargs):
    return _GridSearchCV(estimator, build_param_grid(estimator), **kwargs)
