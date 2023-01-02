import pandas as pd
import numpy as np
import gc

from tqdm import tqdm
from datetime import datetime
from modules.processing.base_preprocessing import remove_bad_variables


def preproc_1_1(df, columns, fillna_val=-1):
    df = remove_bad_variables(df)

    preproc_df = df[columns]
    preproc_df = preproc_df.fillna(fillna_val)

    return preproc_df


def preproc_1_4_exp(df, base_features,
                use_sand_feature=False,
                use_steam_inj_features=False,
                use_move_sinj_move=False,
                verbosity=True):
    columns = []
    # get num of steam injectors
    if use_steam_inj_features:
        df['steam_inj_nums'] = df.apply(get_sinjectors_num, axis=1)
        columns += ['steam_inj_nums']

    # move steam injector info columns
    if use_move_sinj_move:
        df = move_sinjectors_columns(df, verbosity=verbosity)

    # ohe preproc. feature
    if use_sand_feature:
        df_ohe = pd.get_dummies(data=df['SAND'], prefix='SAND')
        df = pd.concat([df, df_ohe], axis=1)
        columns += list(df_ohe.columns)

    columns += base_features
    df = preproc_1_1(df, columns=columns, fillna_val=-1)
    return df


def preproc_1_5_exp(df, base_features, add_sandsnum_f=False, verbosity=True):
    df['SURV_DTE'] = df['SURV_DTE'].map(get_surv_dte)

    # new features
    prep_df = get_seq_num(df, res_feature_name='seq_num_un', normalize=False)
    prep_df = get_life_time(prep_df, group=['CMPL_FAC_ID'], res_feature_name='fcl_life_time', normalize=True)
    prep_df = get_life_time(prep_df, group=['CMPL_FAC_ID', 'SAND'], res_feature_name='fcl_sand_life_time', normalize=True)
    new_time_features = ['seq_num_un', 'fcl_life_time', 'fcl_sand_life_time']

    if add_sandsnum_f:
        prep_df['sands_num'] = prep_df['CMPL_FAC_ID'].map(get_sand_num_mapping(df))
        new_time_features += ['sands_num']

    prep_df = preproc_1_4_exp(prep_df, base_features=base_features + new_time_features,
                              use_sand_feature=True, use_steam_inj_features=True,
                              use_move_sinj_move=True, verbosity=verbosity)

    return prep_df


def preproc_1_6_exp(df, base_features, verbosity=True):
    df['SURV_DTE'] = df['SURV_DTE'].map(get_surv_dte)

    # new features
    prep_df = get_seq_num(df, res_feature_name='seq_num_un', normalize=False)
    prep_df = get_life_time(prep_df, group=['CMPL_FAC_ID'], res_feature_name='fcl_life_time', normalize=True)
    prep_df = get_life_time(prep_df, group=['CMPL_FAC_ID', 'SAND'], res_feature_name='fcl_sand_life_time', normalize=True)
    new_features = ['seq_num_un', 'fcl_life_time', 'fcl_sand_life_time']

    # get num of steam injectors
    prep_df['steam_inj_nums'] = prep_df.apply(get_sinjectors_num, axis=1)
    new_features += ['steam_inj_nums']

    # move steam injector info columns
    prep_df = move_sinjectors_columns(prep_df, verbosity=verbosity)

    # ohe sand preproc. feature
    grn = prep_df['SAND'].map(sand_group_num_encoding).to_numpy()
    prep_df['SAND_group'] = [_[0] for _ in grn]
    prep_df['SAND_num'] = [_[1] for _ in grn]

    df_gr_dum = pd.get_dummies(prep_df['SAND_group'], prefix='SAND_group')
    df_num_dum = pd.get_dummies(prep_df['SAND_num'], prefix='SAND_num')
    prep_df = pd.concat([prep_df, df_gr_dum, df_num_dum], axis=1)
    new_features += list(df_gr_dum) + list(df_num_dum)

    preproc_df = preproc_1_1(prep_df, columns=base_features + new_features, fillna_val=-1)
    return preproc_df


# remaked version of preproc. function
def preproc_1_7_remake(df, base_features, fillna_value=-1, verbosity=True,
                       add_norm_total_inj=False,
                       add_norm_stm_inj=False,
                       add_avg_mean_std=False,
                       add_sand_groups=False,
                       add_date=False,
                       add_groups_num=False,
                       add_fchange_features=False,
                       add_mean_sand_group=False,
                       add_norm_time=False):
    prep_df = remove_bad_variables(df)
    new_features = []

    if add_norm_time:
        prep_df['norm_time'] = prep_df['SURV_DTE'].map(get_norm_days_time)
        new_features += ['norm_time']

    prep_df['SURV_DTE'] = prep_df['SURV_DTE'].map(get_surv_dte)

    # add time based features
    prep_df = get_seq_num(prep_df, res_feature_name='seq_num_un', normalize=False)
    prep_df = get_life_time(prep_df, group=['CMPL_FAC_ID'], res_feature_name='fcl_life_time', normalize=True)
    prep_df = get_life_time(prep_df, group=['CMPL_FAC_ID', 'SAND'], res_feature_name='fcl_sand_life_time', normalize=True)
    new_features += ['seq_num_un', 'fcl_life_time', 'fcl_sand_life_time']

    # add normalized total inj features or not
    if add_norm_total_inj:
        prep_df = get_norm_feature_by_group(prep_df, group=['CMPL_FAC_ID'], feature='TOTAL_INJ',
                                            res_feature_name='total_inj_fac')
        prep_df = get_norm_feature_by_group(prep_df, group=['CMPL_FAC_ID', 'SAND'], feature='TOTAL_GNTL_INJ',
                                           res_feature_name='total_inj_fac_sand')
        prep_df = get_norm_feature_by_group(prep_df, group=['CMPL_FAC_ID'], feature='TOTAL_PROD',
                                            res_feature_name='total_prod_fac')
        new_features += ['total_inj_fac', 'total_inj_fac_sand', 'total_prod_fac']

    # add normalized SGMT_CUM_STM_INJ_[1/2/3] features or not
    if add_norm_stm_inj:
        prep_df = get_norm_feature_by_group(prep_df, group=['CMPL_FAC_ID', 'SAND'], feature='SGMT_CUM_STM_INJ_1',
                                           res_feature_name='cum_stm1_fac_sand', fillna=-1)
        prep_df = get_norm_feature_by_group(prep_df, group=['CMPL_FAC_ID', 'SAND'], feature='SGMT_CUM_STM_INJ_2',
                                           res_feature_name='cum_stm2_fac_sand', fillna=-1)
        prep_df = get_norm_feature_by_group(prep_df, group=['CMPL_FAC_ID', 'SAND'], feature='SGMT_CUM_STM_INJ_3',
                                           res_feature_name='cum_stm3_fac_sand', fillna=-1)
        new_features += ['cum_stm1_fac_sand', 'cum_stm2_fac_sand', 'cum_stm3_fac_sand']

    # add facility/facility+sand averaged features
    if add_avg_mean_std:
        OIL_SAT_mean_map, OIL_SAT_std_map = get_mean_std_feature(prep_df, feature='AVG_ORIG_OIL_SAT')
        prep_df['OIL_SAT_m'] = prep_df['CMPL_FAC_ID'].map(OIL_SAT_mean_map)
        prep_df['OIL_SAT_s'] = prep_df['CMPL_FAC_ID'].map(OIL_SAT_std_map)
        new_features += ['OIL_SAT_m', 'OIL_SAT_s']

    if add_sand_groups:
        grn = prep_df['SAND'].map(sand_group_num_encoding).to_numpy()
        prep_df['SAND_group'] = [_[0] for _ in grn]
        prep_df['SAND_num'] = [_[1] for _ in grn]

        df_gr_dum = pd.get_dummies(prep_df['SAND_group'], prefix='SAND_group')
        df_num_dum = pd.get_dummies(prep_df['SAND_num'], prefix='SAND_num')
        prep_df = pd.concat([prep_df, df_gr_dum, df_num_dum], axis=1)
        new_features += list(df_gr_dum.columns) + list(df_num_dum.columns)

    if add_date:
        prep_df['year'] = prep_df['SURV_DTE'].map(get_year)
        new_features += ['year']

    if add_groups_num:
        grn = prep_df['SAND'].map(sand_group_num_encoding).to_numpy()
        prep_df['SAND_group'] = [_[0] for _ in grn]
        prep_df['SAND_num'] = [_[1] for _ in grn]

        gr_map = prep_df.groupby(by='CMPL_FAC_ID')['SAND_group'].nunique().tolist()
        gr_map = dict(zip(prep_df['CMPL_FAC_ID'].unique().tolist(), gr_map))
        prep_df['groups_num'] = prep_df['CMPL_FAC_ID'].map(gr_map)
        new_features += ['groups_num']

    if add_fchange_features:
        prep_df = get_feature_change_by_time(prep_df, feature='TOTAL_INJ', new_feature_name='total_inj_change')
        new_features += ['total_inj_change']

    if add_mean_sand_group:
        grn = prep_df['SAND'].map(sand_group_num_encoding).to_numpy()
        prep_df['SAND_group'] = [_[0] for _ in grn]

        prep_df = get_mean_osat(prep_df, sgroup_col='SAND_group')
        prep_df = get_sinj_by_sand_group(prep_df, sgroup_col='SAND_group')

        new_features += ['mean_group_orig_sat', 'mean_group_sinj_gntl']





    # get num of steam injectors
    prep_df['steam_inj_nums'] = prep_df.apply(get_sinjectors_num, axis=1)
    new_features += ['steam_inj_nums']

    # move steam injector info columns
    prep_df = move_sinjectors_columns(prep_df, verbosity=verbosity)

    # add sand features
    df_ohe = pd.get_dummies(data=prep_df['SAND'], prefix='SAND')
    prep_df = pd.concat([prep_df, df_ohe], axis=1)
    new_features += list(df_ohe.columns)

    features = base_features + new_features
    prep_df = prep_df[features]
    prep_df = prep_df.fillna(fillna_value)

    return prep_df


def get_norm_days_time(date, min_date='1990.01.01', max_date='2022.01.01'):
    min_date, max_date = pd.to_datetime(min_date), pd.to_datetime(max_date)
    max_days = (max_date - min_date).days

    d = pd.to_datetime(date)

    return (d - min_date).days / max_days


def get_sinj_by_sand_group(df, sgroup_col='sand_group', new_feature_name='mean_group_sinj_gntl'):
    gr = df.groupby(by=['CMPL_FAC_ID', sgroup_col, 'SURV_DTE'])['TOTAL_GNTL_INJ'].mean()

    gr = gr.unstack(level=2)

    f = lambda x: gr.loc[x['CMPL_FAC_ID']].loc[x[sgroup_col]].loc[x['SURV_DTE']]
    df[new_feature_name] = df.apply(f, axis=1)

    return df


def get_mean_osat(df, sgroup_col='sand_group', new_feature_name='mean_group_orig_sat'):
    gr = df.groupby(by=['CMPL_FAC_ID', sgroup_col])['AVG_ORIG_OIL_SAT'].mean()
    gr = gr.unstack(level=1)
    f = lambda x: gr.loc[x['CMPL_FAC_ID'], x[sgroup_col]]
    df[new_feature_name] = df.apply(f, axis=1)

    return df


def get_feature_change_by_time(df, feature, new_feature_name, normalize=False):
    indexes, values = [], []
    for mini_df in df.sort_values(by='SURV_DTE').groupby(by=['CMPL_FAC_ID', 'SAND']):
        mini_df = mini_df[1].reset_index()
        changes = mini_df[feature].diff()

        if normalize:
            changes = changes / mini_df[feature].shift(periods=1)
            changes = changes.replace(np.inf, 1.0)

        changes = changes.fillna(0)

        indexes += mini_df['index'].tolist()
        values += changes.tolist()

    df.loc[indexes, new_feature_name] = values
    return df


def get_year(d):

    d = pd.to_datetime(d)
    return d.year


# for dip, orig_oil_sat, orig_oil_h
def get_mean_std_feature(df, feature):
    df_dupl = df.drop_duplicates(subset=['CMPL_FAC_ID', 'SAND'])
    _mean = df_dupl.groupby(by='CMPL_FAC_ID')[feature].mean()
    _std = df_dupl.groupby(by='CMPL_FAC_ID')[feature].std()

    del df_dupl
    gc.collect()
    _mean, _std = _mean.to_dict(), _std.to_dict()
    return _mean, _std


def get_norm_feature_by_group(df, group=['CMPL_FAC_ID'], feature='TOTAL_INJ', res_feature_name='total_inj_fac',
                              fillna=0):

    index, values = [], []
    for mini_df in df.groupby(by=group):
        mini_df = mini_df[1].reset_index()
        max_toinj = mini_df[feature].max()

        mini_df[feature] = mini_df[feature] / max_toinj
        mini_df[feature] = mini_df[feature].replace(np.nan, fillna)
        # print(mini_df)

        index += mini_df['index'].to_list()
        values += mini_df[feature].to_list()

    # print(values)
    df.loc[index, res_feature_name] = values
    df[res_feature_name] = df[res_feature_name].astype(float)
    return df


def sand_group_num_encoding(sand_id):
    if 'B' in sand_id:
        group = 1
    elif 'D' in sand_id:
        group = 2
    elif 'C' in sand_id:
        group = 3
    else:
        group = 4
    num = sand_id[-1]

    return int(group), int(num)


def get_sand_num_mapping(df):
    sand_num_map = dict()
    for i in list(df['CMPL_FAC_ID'].unique()):
        sand_num_map[i] = len(df[df['CMPL_FAC_ID'] == i]['SAND'].unique())

    return sand_num_map


def get_surv_dte(s):
    if '/' in s:
        return datetime.strptime(s, '%m/%d/%Y')
    else:
        return datetime.strptime(s, '%Y-%m-%d')


# relationship between normalzied seq_num and target
def get_seq_num(df, res_feature_name='seq_num', normalize=True):
    df = df.copy()
    df = df.sort_values(by='SURV_DTE')

    group = df.groupby(by=['CMPL_FAC_ID', 'SAND'])
    group_camcount = group.cumcount()
    df[res_feature_name] = df.index.map(group_camcount)

    if normalize:
        sand_count_mapping = dict()
        for i in df['CMPL_FAC_ID'].unique():
            _df = df[df['CMPL_FAC_ID'] == i]
            sand_count_mapping[i] = _df.groupby(by='SAND')[res_feature_name].max().to_dict()
        df[res_feature_name] = df[res_feature_name] / df.apply(
            lambda x: sand_count_mapping[x['CMPL_FAC_ID']][x['SAND']], axis=1)
        df[res_feature_name] = df[res_feature_name].fillna(0)

    df[res_feature_name] = df[res_feature_name].astype(float)
    df = df.sort_index()
    return df


def get_life_time(df, group=['CMPL_FAC_ID'], res_feature_name='fcl_life_time', normalize=True):
    df = df.copy()
    df[res_feature_name] = df['SURV_DTE']

    index, values = [], []
    for mini_df in df.groupby(by=group):
        mini_df = mini_df[1].reset_index()
        min_lifetime = mini_df[res_feature_name].min()
        mini_df[res_feature_name] = mini_df[res_feature_name].map(lambda x: (x - min_lifetime).days)
        mini_df[res_feature_name] = mini_df[res_feature_name].replace(0, 1)

        if normalize:
            mini_df[res_feature_name] = (mini_df[res_feature_name]) / mini_df[res_feature_name].max()

        index += mini_df['index'].to_list()
        values += mini_df[res_feature_name].to_list()

    # print(values)
    df.loc[index, res_feature_name] = values
    df[res_feature_name] = df[res_feature_name].astype(float)
    return df


def get_first_fcl_month(df, res_feature_name='first_fcl_month'):
    df = df.copy()
    df[res_feature_name] = df['SURV_DTE']

    fm_mapping = df.groupby(by='CMPL_FAC_ID')[res_feature_name].min().map(lambda x: x.month)
    df[res_feature_name] = df['CMPL_FAC_ID'].map(fm_mapping)
    df[res_feature_name] = df[res_feature_name].astype(float)
    return df


def move_sinjectors_columns(df, verbosity=True):
    df = df.copy()
    steam_inj_cols = [[f'SGMT_CUM_STM_INJ_{i + 1}', f'FT_DIST_PAT_{i + 1}'] for i in range(3)]
    for i, r in tqdm(df.iterrows(), desc='Move steam injectors columns...', disable=not verbosity):
        # get all not nan SGMT_CUM_STM_INJ and FT_DIST_PAT features
        st_v, st_d = [], []
        for st_i in steam_inj_cols:
            # all features of one steam inj is not nan
            if not (pd.isna(r[st_i[0]]) or pd.isna(r[st_i[1]])):
                st_v.append(r[st_i[0]])
                st_d.append(r[st_i[1]])

        # fill empty nan values
        nan_num = 3 - len(st_v)
        st_v = st_v + [np.nan] * nan_num
        st_d = st_d + [np.nan] * nan_num

        df.loc[i, [sg[0] for sg in steam_inj_cols]] = st_v  # steam_inj_features
        df.loc[i, [sg[1] for sg in steam_inj_cols]] = st_d
    return df


def get_sinjectors_num(r):
    st_v, st_d = [], []
    st_v.append(r['SGMT_CUM_STM_INJ_1'])
    st_d.append(r['FT_DIST_PAT_1'])
    st_v.append(r['SGMT_CUM_STM_INJ_2'])
    st_d.append(r['FT_DIST_PAT_2'])
    st_v.append(r['SGMT_CUM_STM_INJ_3'])
    st_d.append(r['FT_DIST_PAT_3'])

    inj_num = 0
    for i in range(3):
        if not (pd.isna(st_v[i]) or pd.isna(st_d[i])):
            inj_num += 1

    return inj_num