import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, GradientBoostingRegressor
from modules.processing.experiment_preprocessings import preproc_1_7_remake
from modules.scoring.cross_val_score import get_cv_rmse_score_1


if __name__ == '__main__':
    # model params
    random_state = 42
    CV = 4
    n_trees = 200#3000 #
    criterion = 'squared_error'

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    train_df_path = PROJ_PATH + "data/raw/train.csv"

    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]
    group_col = 'CMPL_FAC_ID'
    target_col = 'PCT_DESAT_TO_ORIG'
    time_col = 'SURV_DTE'

    df = pd.read_csv(train_df_path)

    preproc_df = preproc_1_7_remake(df, base_features=needest_columns + [target_col] + [group_col] + [time_col],
                                    add_norm_total_inj=True,# 1.7.1
                                    add_norm_stm_inj=True,# 1.7.2
                                    add_sand_groups=True,# 1.7.3
                                    add_mean_sand_group=True, # 1.7.6
                                    add_groups_num=True) # 1.7.8

    # get meedest columns
    needest_sand_columns = [c for c in preproc_df.columns if 'SAND_' in c]
    needest_columns = needest_columns + \
                      needest_sand_columns + \
                      ['steam_inj_nums'] + \
                      ['seq_num_un', 'fcl_life_time', 'fcl_sand_life_time'] + \
                      ['total_inj_fac', 'total_inj_fac_sand', 'total_prod_fac'] + \
                      ['cum_stm1_fac_sand', 'cum_stm2_fac_sand', 'cum_stm3_fac_sand'] + \
                      ['mean_group_orig_sat', 'mean_group_sinj_gntl'] + \
                      ['groups_num']


    # test model
    model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion, n_jobs=-1)

    # backward feature elimination
    #needest_columns = needest_columns[:4]
    min_features = 25
    bad_features, logs = [], []
    for i in tqdm(range(len(needest_columns) - min_features)):
        present_features = [i for i in needest_columns if i not in bad_features]
        features_errors_buffer = dict()

        for f in tqdm(present_features, desc='Inner loop... ', disable=True):
            _features = [pf for pf in present_features if pf != f]

            feature_score, _ = get_cv_rmse_score_1(model, preproc_df,
                                                x_cols=_features,
                                                y_col=target_col,
                                                group_col=group_col,
                                                cv=CV,
                                                verbosity=False)
            feature_score = sum(feature_score) / len(feature_score)
            features_errors_buffer[f] = feature_score

        bad_feature = min(features_errors_buffer.items(), key=lambda item: item[1])
        logs.append(bad_feature)
        bad_features.append(bad_feature[0])
        print(f"New bad feature: {bad_feature[0]}, {bad_feature[1]}")

    # plot results
    _, ax = plt.subplots()

    x = list(range(len(logs)))
    y = [i[1] for i in logs]

    sns.lineplot(x=x, y=y, ax=ax)
    for i, l in enumerate(logs):
        f, score = l
        ax.text(i, score, f'{f} \n{round(score, 4)}')

    plt.show()


