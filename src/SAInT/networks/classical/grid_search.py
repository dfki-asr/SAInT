from sklearn.model_selection import GridSearchCV
from timeit import default_timer as timer


def do_grid_search(model, search_space):
    start_time = timer()
    clf = GridSearchCV(estimator=model.model,
                       param_grid=search_space,
                       n_jobs=1,
                       cv=3)
    clf.fit(model.inputs, model.outputs)
    elapsed_time = timer() - start_time
    print(
        f"Grid search for model {model.name} took {elapsed_time:.2f} seconds.")
    print(f"Score: {clf.best_score_:.3f}")
    print(f"Best parameters: {clf.best_params_}")
    return clf
