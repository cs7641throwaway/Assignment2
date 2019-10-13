import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import ast
import math


def read_data(problem, algo, normalize=False):
    # Takes the input, converts string to list, explodes lists to form create individual rows for each type
    file = 'ABAGAIL/jython/output/'+algo+'_'+problem+'_results.csv'
    # Read file into data frame
    # Calculate median and sigma for both scores and runtimes in df
    # Return dataframe
    #df = pd.read_csv(file)
    df = pd.read_csv(file, quotechar='"', converters={1:ast.literal_eval})
    #print(df.shape)
    #print(df.head(5))
    # Convert string to list
    df['scores'] = df['scores'].apply(lambda x: ast.literal_eval(x))
    df['runtimes'] = df['runtimes'].apply(lambda x: ast.literal_eval(x))
    # Explode columns
    df2 = explode(df, ['scores', 'runtimes'])
    #print(df2.shape)
    #print(df2.head(5))
    # mean & sigma  & merge
    cols = df2.columns.tolist()
    cols.remove('scores')
    cols.remove('runtimes')
    df3 = df2.groupby(cols, as_index=False).mean()
    df4 = df2.groupby(cols, as_index=False).std()
    df3['scores_mean'] = df3['scores']
    df3['runtimes_mean'] = df3['runtimes']
    df3['scores_sigma'] = df4['scores']
    df3['runtimes_sigma'] = df4['runtimes']
    df3 = df3.drop(columns=['scores', 'runtimes'])
    if normalize:
        df3['scores_mean'] = df3['scores_mean']/df3['num_points']
        df3['scores_sigma'] = df3['scores_sigma'] / df3['num_points']
    return df3


# No mean scores, since there's only one iteration
def read_data_nn(algo, normalize=False):
    # Takes the input, converts string to list, explodes lists to form create individual rows for each type
    file = 'ABAGAIL/jython/output/' + algo + '_nn_results.csv'
    # Read file into data frame
    # Calculate median and sigma for both scores and runtimes in df
    # Return dataframe
    # df = pd.read_csv(file)
    df = pd.read_csv(file, quotechar='"', converters={1: ast.literal_eval})
    return df


def get_best_param(df):
    row_index = df['scores_mean'].idxmax()
    return df.loc[row_index]

def get_best_param_nn(df):
    row_index = df['training_accuracy'].idxmax()
    return df.loc[row_index]

def plot_param(df, y, x, x_label, y_label, title, opt=True):
    # print(df.head(5))
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    if opt:
        y_mean = y+'_mean'
        y_sigma = y+'_sigma'
        plt.fill_between(df[x], df[y_mean] - df[y_sigma],
                     df[y_mean] + df[y_sigma], alpha=0.1,
                     color="r")
        plt.plot(df[x], df[y_mean], 'o-', color="g")
    else:
        plt.plot(df[x], df[y], 'o-', color="g")
    # plt.legend(loc="best")
    plt.savefig(title+'.png')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = ms.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=1, shuffle = True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    #plt.show()
    plt.savefig(title+'.png')
    return plt


def perform_grid_search(estimator, type, dataset, params, trg_X, trg_Y, tst_X, tst_Y, cv=5, n_jobs=-1, train_score=True):
    cv = ms.GridSearchCV(estimator, n_jobs=n_jobs, param_grid= params, refit=True, verbose=2, cv=cv, return_train_score=train_score)
    cv.fit(trg_X, trg_Y)
    test_score = cv.score(tst_X, tst_Y)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./results/{}_{}_reg.csv'.format(type,dataset),index=False)
    with open('./results/test_results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(type,dataset,test_score,cv.best_params_))


def plot_distribution():
    data = pd.read_hdf('datasets_full.hdf', 'fmnist')
    d = data['Class'].value_counts()
    d.plot.bar()
    plt.ylabel('Frequency')
    plt.xlabel('Class')
    plt.title('FMNIST Class Distribution')
    plt.savefig('FMNIST_class_dist.png')


def plot_complexity(title, param_str, df, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_str)
    plt.ylabel("Score")
    plt.grid()

    # TODO: Need data in form of param, value, training mean, training std, validation mean, validation std
    param_values = df['param_'+param_str]
    train_scores_mean = df['mean_train_score']
    train_scores_std = df['std_train_score']
    test_scores_mean = df['mean_test_score']
    test_scores_std = df['std_test_score']


    plt.fill_between(param_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(param_values, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_values, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(param_values, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    #plt.show()
    plt.savefig(title+'.png')
    return plt


def get_model_complexity_data(file, param=None, param_value=None):
    df = pd.read_csv(file)
    if param is None:
        return df
    df2 = df[df['param_'+param] == param_value]
    # Open file
    # Read into df
    # Slice to get df
    # Format accordingly
    return df2

def plot_DT_complexity():
    file = 'results/DT_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_max_depth']==20] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==5] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_max_depth=20_min_samples_leaf=5", 'min_impurity_decrease', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==5] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_min_impurity_decr=0.0005_min_samples_leaf=5", 'max_depth', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_max_depth']==20] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_min_impurity_decr=0.0005_max_depth=20", 'min_samples_leaf', df)
    file = 'results/DT_chess_reg.csv'
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_max_depth']==100] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==1] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_max_depth=100_min_samples_leaf=1", 'min_impurity_decrease', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==1] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_min_impurity_decr=0.0005_min_samples_leaf=1", 'max_depth', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_max_depth']==100] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_min_impurity_decr=0.0005_max_depth=100", 'min_samples_leaf', df)

def plot_SVM_complexity():
    file = 'results/SVM_Linear_chess_reg.csv'
    df = get_model_complexity_data(file, 'penalty', 'l1')
    plot_complexity("chess_SVM_Linear_alpha", 'alpha', df)
    file = 'results/SVM_RBF_chess_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("chess_SVM_RBF_C", 'C', df)
    file = 'results/SVM_Linear_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'penalty', 'l1')
    plot_complexity("FMNIST_SVM_Linear_alpha", 'alpha', df)
    file = 'results/SVM_RBF_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_C", 'C', df)

def plot_kNN_complexity():
    file = 'results/kNN_chess_reg.csv'
    df = get_model_complexity_data(file, 'weights', 'uniform')
    plot_complexity("chess_kNN_uniform_n_neighbors", 'n_neighbors', df)
    df = get_model_complexity_data(file, 'weights', 'distance')
    plot_complexity("chess_kNN_distance_n_neighbors", 'n_neighbors', df)
    file = 'results/kNN_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'weights', 'uniform')
    plot_complexity("FMNIST_kNN_uniform_n_neighbors", 'n_neighbors', df)
    df = get_model_complexity_data(file, 'weights', 'distance')
    plot_complexity("FMNIST_kNN_distance_n_neighbors", 'n_neighbors', df)

def plot_NN_complexity():
    file = 'results/NN_chess_reg.csv'
    df = get_model_complexity_data(file, 'hidden_layer_sizes', "(50, 10)")
    plot_complexity("chess_NN_relu_hidden_layers=(50,10)_tol", 'tol', df)
    file = 'results/NN_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'tol', 10**-3)
    plot_complexity("FMNIST_NN_tol_10e-3_hidden_layers", 'hidden_layer_sizes', df)
    df = get_model_complexity_data(file, 'hidden_layer_sizes', "(50,)")
    plot_complexity("FMNIST_NN_hidden_layers_50_tol", 'tol', df)

def plot_NN_iteration_curve():
    file = 'results/NN_iteration_curve_chess_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("chess_NN_iteration_curve", 'max_iter', df)
    file = 'results/NN_iteration_curve_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_NN_iteration_curve", 'max_iter', df)

def plot_SVM_iteration_curve():
    file = 'results/SVM_RBF_iteration_curve_chess_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("chess_SVM_RBF_iteration_curve", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_300_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve_300", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_500_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve_500", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_1000_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve_1000", 'max_iter', df)
    file = 'results/SVM_RBF_iteration_curve_2000_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_iteration_curve_2000", 'max_iter', df)

def plot_boosting_complexity():
    file = 'results/boosting_chess_reg.csv'
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'random')
    df2 = df[df['param_n_estimators'] == 200]
    plot_complexity('chess_boosting_splitter_random_n_estimators_200_max_depth', 'base_estimator__max_depth', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'random')
    df2 = df[df['param_base_estimator__max_depth'] == 5]
    plot_complexity('chess_boosting_splitter_random_max_depth_5_n_estimators', 'n_estimators', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'best')
    df2 = df[df['param_n_estimators'] == 200]
    plot_complexity('chess_boosting_splitter_best_n_estimators_200_max_depth', 'base_estimator__max_depth', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'best')
    df2 = df[df['param_base_estimator__max_depth'] == 5]
    plot_complexity('chess_boosting_splitter_best_max_depth_5_n_estimators', 'n_estimators', df2)
    file = 'results/boosting_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'n_estimators', 100)
    plot_complexity('FMNIST_boosting_splitter_random_100_estimators_max_depth', 'base_estimator__max_depth', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 20)
    plot_complexity('FMNIST_boosting_splitter_random_max_depth_20_n_estimators', 'n_estimators', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 10)
    plot_complexity('FMNIST_boosting_splitter_random_max_depth_10_n_estimators', 'n_estimators', df)
    file = 'results/boosting_FMNIST_best_reg.csv'
    df = get_model_complexity_data(file, 'n_estimators', 100)
    plot_complexity('FMNIST_boosting_splitter_best_100_estimators_max_depth', 'base_estimator__max_depth', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 20)
    plot_complexity('FMNIST_boosting_splitter_best_max_depth_20_n_estimators', 'n_estimators', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 10)
    plot_complexity('FMNIST_boosting_splitter_best_max_depth_10_n_estimators', 'n_estimators', df)

# Data to pull
    # Best mean for each of the algos
    # Average runtime for that mean
# Plots to make
    # Score vs. Iteration Count
    # Score vs. Problem size

# Stack overflow
def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res


def get_best_params(problem):
    print("Best selections for ", problem, ": ")
    df_rhc = read_data(problem, "rhc")
    slice = get_best_param(df_rhc)
    print("RHC: ", slice[['scores_mean', 'runtimes_mean']])

    df_sa = read_data(problem, "sa")
    slice = get_best_param(df_sa)
    print("SA: ", slice[['cooling', 'temp', 'scores_mean', 'runtimes_mean']])

    df_ga = read_data(problem, "ga")
    slice = get_best_param(df_ga)
    print("GA: ", slice[['pop', 'prop_mate', 'prop_mutate', 'scores_mean', 'runtimes_mean']])

    df_mi = read_data(problem, "mi")
    slice = get_best_param(df_mi)
    print("MIMIC: ", slice[['samples', 'prop_keep', 'scores_mean', 'runtimes_mean']])

def get_best_params_nn():
    print("Best selections for Neural Network based on training score")
    df_rhc = read_data_nn("RHC")
    slice = get_best_param_nn(df_rhc)
    print("RHC: ", slice[['error', 'training_accuracy', 'testing_accuracy', 'runtime']])

    df_sa = read_data_nn("SA_sweep")
    #print(df_sa)
    slice = get_best_param_nn(df_sa)
    #print(slice)
    print("SA: ", slice[['cooling', 'temperature', 'error', 'training_accuracy', 'testing_accuracy', 'runtime']])

    df_ga = read_data_nn("GA_sweep")
    slice = get_best_param_nn(df_ga)
    print("GA: ", slice[['pop', 'prop_mate', 'prop_mutate', 'error', 'training_accuracy', 'testing_accuracy', 'runtime']])

    df_bp = read_data_nn("BP")
    slice = get_best_param_nn(df_bp)
    print("Backpropagation: ", slice[['error', 'training_accuracy', 'testing_accuracy', 'runtime']])


def plot_param_sweep(temp, cooling, pop, prop_mate, prop_mutate, samples, prop_keep, problem):
    df_sa = read_data(problem, "sa")
    df_temp = df_sa[abs(df_sa['cooling']-cooling) < 1E-9]
    plot_param(df=df_temp, x='temp', y='scores', x_label='Temperature', y_label='Score',
               title='SA_'+problem+'_with_Cooling='+str(cooling))
    df_cooling = df_sa[abs(df_sa['temp']-temp) < 1E-9]
    plot_param(df=df_cooling, x='cooling', y='scores', x_label='Cooling', y_label='Score',
               title='SA_'+problem+'_with_Temp={:.0e}'.format(temp))

    df_ga = read_data(problem, "ga")
    df_pop = df_ga[abs(df_ga['prop_mate']-prop_mate) < 1E-9]
    df_pop = df_pop[abs(df_pop['prop_mutate']-prop_mutate) < 1E-9]
    plot_param(df=df_pop, x='pop', y='scores', x_label='Population Size', y_label='Score',
               title='GA_'+problem+'_with_prop_mate='+str(prop_mate)+'_prop_mutate='+str(prop_mutate))
    df_mate = df_ga[abs(df_ga['pop']-pop) < 1E-9]
    df_mate = df_mate[abs(df_mate['prop_mutate']-prop_mutate) < 1E-9]
    plot_param(df=df_mate, x='prop_mate', y='scores', x_label='Proportion of Mate', y_label='Score',
               title='GA_'+problem+'_with_pop='+str(pop)+'_prop_mutate='+str(prop_mutate))
    df_mutate = df_ga[abs(df_ga['pop']-pop) < 1E-9]
    df_mutate = df_mutate[abs(df_mutate['prop_mate']-prop_mate) < 1E-9]
    plot_param(df=df_mutate, x='prop_mutate', y='scores', x_label='Proportion of Mutate', y_label='Score',
               title='GA_'+problem+'_with_pop='+str(pop)+'_prop_mate='+str(prop_mate))

    df_mi = read_data(problem, "mi")
    df_samples = df_mi[abs(df_mi['prop_keep']-prop_keep) < 1E-9]
    plot_param(df=df_samples, x='samples', y='scores', x_label='Number of Samples', y_label='Score',
               title='MIMIC_'+problem+'_with_prop_keep='+str(prop_keep))
    df_keep = df_mi[abs(df_mi['samples']-samples) < 1E-9]
    plot_param(df=df_keep, x='prop_keep', y='scores', x_label='Proportion to Keep', y_label='Score',
               title='MIMIC_'+problem+'_with_samples='+str(samples))


def plot_param_sweep_nn(temp, cooling, pop, prop_mate, prop_mutate):
    # TODO: Need to generate training curves (no validation)
    df_sa = read_data_nn("SA_sweep")
    max_iters = df_sa['iter_num'].max()
    df_sa = df_sa[df_sa['iter_num'] == max_iters]
    df_temp = df_sa[abs(df_sa['cooling']-cooling) < 1E-9]
    plot_param(df=df_temp, x='temperature', y='training_accuracy', x_label='Temperature', y_label='Training Accuracy',
               title='NN_SA_with_Cooling='+str(cooling), opt=False)
    df_cooling = df_sa[abs(df_sa['temperature']-temp) < 1E-9]
    plot_param(df=df_cooling, x='cooling', y='training_accuracy', x_label='Cooling', y_label='Training Accuracy',
               title='NN_SA_with_Temp={:.0e}'.format(temp), opt=False)

    df_ga = read_data_nn("GA_sweep")
    max_iters = df_ga['iter_num'].max()
    df_ga = df_ga[df_ga['iter_num'] == max_iters]
    df_pop = df_ga[abs(df_ga['prop_mate']-prop_mate) < 1E-9]
    df_pop = df_pop[abs(df_pop['prop_mutate']-prop_mutate) < 1E-9]
    plot_param(df=df_pop, x='pop', y='training_accuracy', x_label='Population Size', y_label='Training Accuracy',
               title='NN_GA_with_prop_mate='+str(prop_mate)+'_prop_mutate='+str(prop_mutate), opt=False)
    df_mate = df_ga[abs(df_ga['pop']-pop) < 1E-9]
    df_mate = df_mate[abs(df_mate['prop_mutate']-prop_mutate) < 1E-9]
    plot_param(df=df_mate, x='prop_mate', y='training_accuracy', x_label='Proportion of Mate',
               y_label='Training Accuracy', title='NN_GA_with_pop='+str(pop)+'_prop_mutate='+str(prop_mutate),
               opt=False)
    df_mutate = df_ga[abs(df_ga['pop']-pop) < 1E-9]
    df_mutate = df_mutate[abs(df_mutate['prop_mate']-prop_mate) < 1E-9]
    plot_param(df=df_mutate, x='prop_mutate', y='training_accuracy', x_label='Proportion of Mutate',
               y_label='Training Accuracy', title='NN_GA_with_pop='+str(pop)+'_prop_mate='+str(prop_mate), opt=False)


def plot_input_size_sweep(problem, normalize_scores=False):
    str = problem+'_size'
    df_rhc = read_data(str, "rhc", normalize_scores)
    plot_param(df=df_rhc, x='num_points', y='scores', x_label='Number of Points', y_label='Score',
                title='RHC_'+problem+'_input_size_score')
    plot_param(df=df_rhc, x='num_points', y='runtimes', x_label='Number of Points', y_label='Runtime',
               title='RHC_'+problem+'_input_size_runtime')

    df_sa = read_data(str, "sa", normalize_scores)
    plot_param(df=df_sa, x='num_points', y='scores', x_label='Number of Points', y_label='Score',
               title='SA_'+problem+'_input_size_score')
    plot_param(df=df_sa, x='num_points', y='runtimes', x_label='Number of Points', y_label='Runtime',
               title='SA_'+problem+'_input_size_runtime')

    df_ga = read_data(str, "ga", normalize_scores)
    plot_param(df=df_ga, x='num_points', y='scores', x_label='Number of Points', y_label='Score',
               title='GA_'+problem+'_input_size_score')
    plot_param(df=df_ga, x='num_points', y='runtimes', x_label='Number of Points', y_label='Runtime',
               title='GA_'+problem+'_input_size_runtime')

    df_mi = read_data(str, "mi", normalize_scores)
    plot_param(df=df_mi, x='num_points', y='scores', x_label='Number of Points', y_label='Score',
               title='MIMIC_'+problem+'_input_size_score')
    plot_param(df=df_mi, x='num_points', y='runtimes', x_label='Number of Points', y_label='Runtime',
               title='MIMIC_'+problem+'_input_size_runtime')

    problem_title = "UNDEFINED"
    if problem == "tsp":
        problem_title = "TSP: Fitness vs. Input Size"
    if problem == "knapsack":
        problem_title = "Knapsack: Fitness vs. Input Size"
    if problem == "continuouspeaks":
        problem_title = "Continuous Peaks: Fitness vs. Input Size"
    if problem == "countones":
        problem_title = "Count Ones: Fitness vs. Input Size"
    plot_all_algos_in_one_plot(df_rhc, df_sa, df_ga, df_mi, x='num_points', y='scores', x_label='Input Size',
                               y_label='Mean Score', title=problem_title,
                               file_name=problem+'_input_size_all_algos.png')
#    plot_all_algos_in_one_plot(df_rhc, df_sa, df_ga, df_mi, x='num_points', y='runtimes', x_label='Input Size',
#                               y_label='Mean Runtime', title=problem_title,
#                               file_name=problem+'_input_size_all_algos_runtime.png')


def plot_all_algos_in_one_plot(df_rhc, df_sa, df_ga, df_mi, x, y, x_label, y_label, title, file_name):
    # WIP
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    y_mean = y + '_mean'
    y_sigma = y + '_sigma'
    plt.plot(df_rhc[x], df_rhc[y_mean], 'o-', color="k", label="RHC")
    plt.plot(df_sa[x], df_sa[y_mean], 's--', color="b", label="SA")
    plt.plot(df_ga[x], df_ga[y_mean], 'D-.', color="r", label="GA")
    plt.plot(df_mi[x], df_mi[y_mean], 'x-.', color="g", label="MIMIC")
    plt.legend(loc="best")
    plt.savefig(file_name)


def plot_all_algos_in_one_plot_nn(df_rhc, df_sa, df_ga, df_bp, x, y, x_label, y_label, title, file_name):
    # WIP
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.plot(df_rhc[x], df_rhc[y], 'o-', color="k", label="RHC")
    plt.plot(df_sa[x], df_sa[y], 's--', color="b", label="SA")
    plt.plot(df_ga[x], df_ga[y], 'D-.', color="r", label="GA")
    plt.plot(df_bp[x], df_bp[y], 'x-.', color="g", label="BP")
    plt.legend(loc="best")
    plt.savefig(file_name)


def plot_iter_sweep(problem, normalize_scores=False):
    str = problem + '_iter'
    df_rhc = read_data(str, "rhc", normalize_scores)
    plot_param(df=df_rhc, x='iter', y='scores', x_label='Number of Iterations', y_label='Score',
               title='RHC_' + problem + '_num_iterations_score')
    plot_param(df=df_rhc, x='iter', y='runtimes', x_label='Number of Iterations', y_label='Runtime',
               title='RHC_' + problem + '_num_iterations_runtime')

    df_sa = read_data(str, "sa", normalize_scores)
    plot_param(df=df_sa, x='iter', y='scores', x_label='Number of Iterations', y_label='Score',
               title='SA_' + problem + '_num_iterations_score')
    plot_param(df=df_sa, x='iter', y='runtimes', x_label='Number of Iterations', y_label='Runtime',
               title='SA_' + problem + '_num_iterations_runtime')

    df_ga = read_data(str, "ga", normalize_scores)
    plot_param(df=df_ga, x='iter', y='scores', x_label='Number of Iterations', y_label='Score',
               title='GA_' + problem + '_num_iterations_score')
    plot_param(df=df_ga, x='iter', y='runtimes', x_label='Number of Iterations', y_label='Runtime',
               title='GA_' + problem + '_num_iterations_runtime')

    df_mi = read_data(str, "mi", normalize_scores)
    plot_param(df=df_mi, x='iter', y='scores', x_label='Number of Iterations', y_label='Score',
               title='MIMIC_' + problem + '_num_iterations_score')
    plot_param(df=df_mi, x='iter', y='runtimes', x_label='Number of Iterations', y_label='Runtime',
               title='MIMIC_' + problem + '_num_iterations_runtime')

    problem_title = "UNDEFINED"
    if problem == "tsp":
        problem_title = "TSP: Runtime vs. Number of Iterations"
    if problem == "knapsack":
        problem_title = "Knapsack: Runtime vs. Number of Iterations"
    if problem == "continuouspeaks":
        problem_title = "Continuous Peaks: Runtime vs. Number of Iterations"
    if problem == "countones":
        problem_title = "Count Ones: Runtime vs. Number of Iterations"
    if 1:
        df_rhc = df_rhc[df_rhc['iter'] < 200000]
        df_sa = df_sa[df_sa['iter'] < 200000]
    df_rhc['runtimes_mean'] = np.log10(df_rhc['runtimes_mean'])
    df_sa['runtimes_mean'] = np.log10(df_sa['runtimes_mean'])
    df_ga['runtimes_mean'] = np.log10(df_ga['runtimes_mean'])
    df_mi['runtimes_mean'] = np.log10(df_mi['runtimes_mean'])
    plot_all_algos_in_one_plot(df_rhc, df_sa, df_ga, df_mi, x='iter', y='runtimes', x_label='Number of Iterations',
                               y_label='Log Mean Runtime', title=problem_title,
                               file_name=problem+'_iter_all_algos_runtime.png')


# Need to slice off of specific parameter values
def plot_iter_sweep_nn(temp, cooling, pop, prop_mate, prop_mutate, normalize_scores=False):
    df_rhc = read_data_nn("RHC")
    plot_param(df=df_rhc, x='iter_num', y='training_accuracy', x_label='Number of Iterations',
               y_label='Training Accuracy', title='NN_RHC_num_iterations_training_accuracy', opt=False)
    plot_param(df=df_rhc, x='iter_num', y='runtime', x_label='Number of Iterations', y_label='Runtime',
               title='NN_RHC_num_iterations_runtime', opt=False)

    df_sa = read_data_nn("SA_sweep")
    df_sa = df_sa[abs(df_sa['cooling']-cooling) < 1E-9]
    df_sa = df_sa[abs(df_sa['temperature']-temp) < 1E-9]
    plot_param(df=df_sa, x='iter_num', y='training_accuracy', x_label='Number of Iterations', y_label='Training Accuracy',
               title='NN_SA_num_iterations_training_accuracy', opt=False)
    plot_param(df=df_sa, x='iter_num', y='runtime', x_label='Number of Iterations', y_label='Runtime',
               title='NN_SA_num_iterations_runtime', opt=False)

    df_ga = read_data_nn("GA_sweep")
    df_ga = df_ga[abs(df_ga['pop']-pop) < 1E-9]
    df_ga = df_ga[abs(df_ga['prop_mate']-prop_mate) < 1E-9]
    df_ga = df_ga[abs(df_ga['prop_mutate']-prop_mutate) < 1E-9]
    plot_param(df=df_ga, x='iter_num', y='training_accuracy', x_label='Number of Iterations', y_label='Training Accuracy',
               title='NN_GA_num_iterations_training_accuracy', opt=False)
    plot_param(df=df_ga, x='iter_num', y='runtime', x_label='Number of Iterations', y_label='Runtime',
               title='NN_GA_num_iterations_runtime', opt=False)

    df_bp = read_data_nn("BP")
    plot_param(df=df_bp, x='iter_num', y='training_accuracy', x_label='Number of Iterations', y_label='Training Accuracy',
               title='NN_BP_num_iterations_training_accuracy', opt=False)
    plot_param(df=df_bp, x='iter_num', y='runtime', x_label='Number of Iterations', y_label='Runtime',
               title='NN_BP_num_iterations_runtime', opt=False)

    problem_title = 'Neural Network: Runtime vs. Number of Iterations'
    if 1:
        df_rhc = df_rhc[df_rhc['iter_num'] < 200000]
        df_sa = df_sa[df_sa['iter_num'] < 200000]
    df_rhc['runtime'] = np.log10(df_rhc['runtime'])
    df_sa['runtime'] = np.log10(df_sa['runtime'])
    df_ga['runtime'] = np.log10(df_ga['runtime'])
    df_bp['runtime'] = np.log10(df_bp['runtime'])
    plot_all_algos_in_one_plot_nn(df_rhc, df_sa, df_ga, df_bp, x='iter_num', y='runtime', x_label='Number of Iterations',
                               y_label='Log Mean Runtime', title=problem_title,
                               file_name='NN_num_iterations_all_algos_runtime.png')
    problem_title = 'Neural Network: Error vs. Number of Iterations'
    plot_all_algos_in_one_plot_nn(df_rhc, df_sa, df_ga, df_bp, x='iter_num', y='error', x_label='Number of Iterations',
                                  y_label='Error', title=problem_title,
                                  file_name='NN_num_iterations_all_algos_error.png')
    problem_title = 'Neural Network: Training Accuracy vs. Number of Iterations'
    plot_all_algos_in_one_plot_nn(df_rhc, df_sa, df_ga, df_bp, x='iter_num', y='training_accuracy',
                                  x_label='Number of Iterations', y_label='Training Accuracy', title=problem_title,
                                  file_name='NN_num_iterations_all_algos_training_accuracy.png')

# Best parameters
#get_best_params("tsp")
# plot_param_sweep(temp=1E13, cooling=0.999, pop=2500, prop_mate=0.85, prop_mutate=0.125, samples=300, prop_keep=0.2,
#                 problem="tsp")
#get_best_params("knapsack")
#plot_param_sweep(temp=1000, cooling=0.99, pop=300, prop_mate=0.85, prop_mutate=0.05, samples=300, prop_keep=0.3,
#                 problem="knapsack")
#get_best_params("continuouspeaks")
# plot_param_sweep(temp=1E10, cooling=0.85, pop=300, prop_mate=0.3, prop_mutate=0.2, samples=100, prop_keep=0.05,
#                 problem="continuouspeaks")
#get_best_params("countones")
# plot_param_sweep(temp=100, cooling=0.85, pop=100, prop_mate=0.9, prop_mutate=0.1, samples=100, prop_keep=0.1,
#                 problem="countones")

# Input sweep
#plot_input_size_sweep("tsp")
#plot_input_size_sweep("knapsack")
plot_input_size_sweep("continuouspeaks")
#plot_input_size_sweep("countones", normalize_scores=True)

# Iteration curve
#plot_iter_sweep("tsp")
#plot_iter_sweep("knapsack")
#plot_iter_sweep("continuouspeaks")
#plot_iter_sweep("countones")

# Neural Network
#get_best_params_nn()
#plot_param_sweep_nn(temp=1E12, cooling=0.95, pop=300, prop_mate=0.5, prop_mutate=0.1)
# Need to plot training accuracy and error as function of iterations
#plot_iter_sweep_nn(temp=1E12, cooling=0.95, pop=300, prop_mate=0.5, prop_mutate=0.1)
