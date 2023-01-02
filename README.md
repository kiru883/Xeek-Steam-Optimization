# Steam Optimization and Other Oddities
This is repository with __very__ raw experiments for [Steam Optimization and Other Oddities challenge](https://xeek.ai/challenges/steam-optimization-and-other-oddities/leaderboard) 
which took [8th place (Kirill D)](https://xeek.ai/challenges/steam-optimization-and-other-oddities/leaderboard) in Final Leaderboard. If you
want to get some data, please, [contact me](https://www.linkedin.com/in/kirud/). 

# How to install
Install virtual environment for python 3.8.
```
pip install -r requirements.txt
```

# Finally solution
Finaly solution is in `/experiments/exp_1.8/train_stacking_exp1_8_1.py` and `/experiments/exp_1.8/get_submission_ad_stacking.py` scripts.


# Experiments
## 1.1 exp.
```
Try predict OIL desaturation using next features:
'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2', 
'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'

Dont use SAND and DATE features. Traing RandomTree model with next params:
n_trees = 3000
criterion = 'squared_error'

Result:
RMSE(Train, test_size=0.2) - 0.2529
RMSE(LB, test_size=0.2) - 0.3478
RMSE(LB, all data) - 0.346
```

## 1.2 exp.
```
All data the same as in 1.1, but train 4 gradient boosting models(by 4 fold cross validation) and predict 
target by 4 models

Training gradient boosting model with next params:
n_trees = 400
lr = 0.1

Result:
RMSE(Train CV4) - 0.2423
RMSE(LB) - 0.3512
```

## 1.3 exp.
```
All data the same as in 1.1(but normalize features), train 4 linear regression 
models(by 4 fold cross validation) and predict target by 4 models.

Training gradient LinReg model with next params:
lr = 0.1

Result:
RMSE(Train CV4) - 0.3699
RMSE(LB) - 0.383
```

## 1.4 exp.
```
All data the same as in 1.1 but add additional features:
1. Number of steam injectors(if SGMT_CUM_STM_INJ_N and FT_DIST_PAT_N columns is not nan,
is steam injector)
2. Move steam injectors columns in left columns side
3. Add SAND OHE feature
4. 1+2+3

Train random forest model.

Training gradient boosting model with next params:
n_trees = 3000

Result:
1:
    RMSE(CV4) - 0.2517 
2:
    RMSE(CV4) - 0.2522
3:
    RMSE(CV4) - 0.2475
4:
    RMSE(CV4) - 0.2478
    RMSE(LB, test_size=0.2) - 0.3398
    RMSE(LB, all data) - 0.338
```


## METRIC.1 exp.
_!!!WARNING!!! NEW CV STRATEGY - GroupKFold(group=CMPL_FAC_ID)_**
```
The estimate was overestimated due to the fact that the model was 
looking for the "oil well - target" relationship instead of the 
"features - target" relationship.....

Train_Test_split without shuffle give me lower RMSE =~-0.34 on exp 1.4....
Try compute all scores on GroupKFold(cv=4), group is oil facility(CMPL_FAC_ID)
Compute scores for this models:

1. RF model from exp 1.1, results
    RMSE(groupkf CV4) - 0.3469
    
2. GradBoost model from exp 1.2
    RMSE(groupkf CV4) - 0.3519

3. LinReg model from exp 1.3
    RMSE(groupkf CV4) - 0.3709
    
4. RF model from exp 1.4, results:
   RMSE(groupkf CV4) -  0.3395

CONCLUSION - this CV strategy high correlated with LB, in next exps i will use 
this CV strategy 
```



## 1.5 exp.
```
Try time features:
plifetime_fcl, plifetime_sand_fcl, lifetime_fcl, lifetime_sand_fcl,
fcl_sand_seq_num, first_life_month

1. Tuned RF model_exp1.4 + normalized fcl_sand_seq_num
    RMSE(groupKF CV4) - 0.3309
    RMSE(LB, trained on all data) - 0.3241
    
2. Tuned RF model_exp1.4 + seq_num_un, fcl_life_time, fcl_sand_life_time
    RMSE(groupKF CV4) - 0.3274
    RMSE(LB, trained on all data) - 0.3244
    
3. As in 2, but add sands_num
    RMSE(groupKF CV4) - 0.3281
    RMSE(LB, trained on all data) - 0.3245
```


## 1.6 exp.
```
Try other method OHE of SAND:
1/4 codes of types of SAND
1/5 codes of number of SAND

Try stacking for best solution

1. 1.5exp Solution + new sand OHE features <- lost .py scripts...
    RMSE(groupKF CV4) - 0.3280
    
2. Stacking with features preprocessing from exp. 1.5
    RF_exp1.5 + SVR + LGBMRegressor, final_estimator - RidgeCV
    RMSE(groupKF CV4) - 0.3258
    RMSE(LB, trained on all data) - 0.3204
```

## 1.7 exp.
```
Add remaining features to exp1.5.2 approach:
1. ADD TOTAL_INJ, TOTAL_GNTL_INJ, TOTAL_PROD on normalized by facility/facility+sand variant
    RMSE(groupKF CV4) - 0.3243
    RMSE(LB, trained on all data) - 0.3222
    
2. ADD SGMT_CUM_STM_INJ_[1/2/3] on normalized by facility/facility+sand variant
    RMSE(groupKF CV4) - 0.3257

3. ADD sand group feature B/C/E/D
    RMSE(groupKF CV4) - 0.3265

4. Mean/STD by facility, facility+SAND features
    RMSE(groupKF CV4) - 0.3273

5. Change of TOTAL_INj, TOTAL_GNTL_INJ, TOTAL_PROD
    RMSE(groupKF CV4) - 0.3272
    
6. Mean ORIG_SAT and TOTAL_GNTL_INJ for each sand group
    RMSE(groupKF CV4) - 0.3260

7. Time feature(year)
    RMSE(groupKF CV4) - 0.3277

8. Number of sand group feature
    RMSE(groupKF CV4) - 0.3265

9. 1 + 2 + 3 + 6 + 8 exps.
    RMSE(groupKF CV4) - 0.3211
    RMSE(LB, trained on all data) - 0.3168
    
10. STACKING data all the same as in 9
    RMSE(groupKF CV4) - 0.32029
    RMSE(LB, trained on all data) - 0.3131
```

## 1.8 exp.
```
Feature selection
1. Permutation importance, WARNING n_trees=300
First - bad features
              Weight               Feature
0    0.0008  ± 0.0012         FT_DIST_PAT_3
1    0.0001  ± 0.0001          SAND_TULS_C3
2    0.0001  ± 0.0010  Lin_Dist_Prod_Factor
3    0.0000  ± 0.0000          SAND_TULS_C1
4    0.0000  ± 0.0003            groups_num
5    0.0000  ± 0.0001        steam_inj_nums
6    0.0000  ± 0.0005     cum_stm2_fac_sand
7   -0.0000  ± 0.0001          SAND_TULS_B3
8   -0.0000  ± 0.0002          SAND_group_4
9   -0.0000  ± 0.0002          SAND_TULS_E3
10  -0.0000  ± 0.0002          SAND_TULS_B2
11  -0.0000  ± 0.0001          SAND_TULS_C4
12  -0.0000  ± 0.0002            SAND_num_4
13  -0.0000  ± 0.0001          SAND_TULS_D5
14  -0.0001  ± 0.0002          SAND_TULS_D4
15  -0.0001  ± 0.0001          SAND_TULS_B5
16  -0.0001  ± 0.0001            SAND_num_3
17  -0.0001  ± 0.0002          SAND_TULS_C5
18  -0.0001  ± 0.0004    fcl_sand_life_time
19  -0.0001  ± 0.0002          SAND_TULS_D3
20  -0.0001  ± 0.0001          SAND_TULS_B4
21  -0.0001  ± 0.0002          SAND_TULS_C2
22  -0.0001  ± 0.0007                   DIP
23  -0.0002  ± 0.0002          SAND_TULS_D2
24  -0.0002  ± 0.0003          SAND_group_3
25  -0.0002  ± 0.0002            SAND_num_2
26  -0.0002  ± 0.0003          SAND_TULS_B1
27  -0.0002  ± 0.0003          SAND_TULS_E1
28  -0.0003  ± 0.0003          SAND_group_1
29  -0.0003  ± 0.0002          SAND_TULS_E4
30  -0.0003  ± 0.0004        total_prod_fac
31  -0.0004  ± 0.0004          SAND_group_2
32  -0.0004  ± 0.0003          SAND_TULS_D1
33  -0.0005  ± 0.0007    SGMT_CUM_STM_INJ_2
34  -0.0006  ± 0.0004          SAND_TULS_E2
35  -0.0006  ± 0.0005            SAND_num_5
36  -0.0007  ± 0.0009         FT_DIST_PAT_2
37  -0.0008  ± 0.0005         total_inj_fac
38  -0.0008  ± 0.0006          SAND_TULS_E5
39  -0.0009  ± 0.0006    SGMT_CUM_STM_INJ_1
40  -0.0012  ± 0.0011   Lin_Dist_Inj_Factor
41  -0.0012  ± 0.0007            SAND_num_1
42  -0.0013  ± 0.0008     cum_stm3_fac_sand
43  -0.0013  ± 0.0006         fcl_life_time
44  -0.0014  ± 0.0010            TOTAL_PROD
45  -0.0021  ± 0.0011    SGMT_CUM_STM_INJ_3
46  -0.0025  ± 0.0016            ORIG_OIL_H
47  -0.0026  ± 0.0014         FT_DIST_PAT_1
48  -0.0031  ± 0.0012        TOTAL_GNTL_INJ
49  -0.0063  ± 0.0018            seq_num_un
50  -0.0065  ± 0.0029   mean_group_orig_sat
51  -0.0075  ± 0.0016     cum_stm1_fac_sand
52  -0.0082  ± 0.0025      AVG_ORIG_OIL_SAT
53  -0.0111  ± 0.0032             TOTAL_INJ
54  -0.0112  ± 0.0016    total_inj_fac_sand
55  -0.0118  ± 0.0027  mean_group_sinj_gntl

2. Choice features using backward features elimination 
    

1.1 Choised 'FT_DIST_PAT_3', 'Lin_Dist_Prod_Factor', 'groups_num', 'steam_inj_nums' features
    RMSE(groupKF CV4) - 0.3189
1.2 1.1 + stacking from 1.7.10 
    RMSE(groupKF CV4) - 0.3173
    RMSE(LB, trained on all data) - 0.3112
```




