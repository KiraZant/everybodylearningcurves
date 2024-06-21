# Added to all feature groups
basic_info = ['start_year', 'Weeks', 'AN', 'Fit', 'Plus', 'basic', 'original']

# Simple questionnaire
wcs_scores = ['screening_wcs', 'wcs_total']

# Additional Extended questionnaire
ext_quest = ['Group chat', 'Feedback', 'Diary',
       'Promote eating & exercise', 'Establish eating & exercise',
       'Body image', 'Self-esteem', 'Occ. binge/ purge', 'Dieatry restrain',
       'Weight regulation', 'age', 'soc_occupational_status_5', 'soc_occupational_status_1',
       'soc_occupational_status_4', 'soc_occupational_status_3',
       'soc_occupational_status_6', 'soc_occupational_status_7',
       'soc_occupational_status_2','soc_education', 'soc_single', 'soc_residency','any_lifetime_disorder_yes_no',
       'any_prior_psychotherapy_yes_no','weightloss_duration', 'weightloss_past_year','nan_personality',
       'bfi_extro', 'bfi_agree', 'bfi_conc', 'bfi_neuro', 'bfi_open','screening_bmi', 'gad7_sum', 'phq9_sum', 'nan_ssrq', 'screening_csed_3m_loss_freq', 'auditc_total_score',
       'bmi','edeq_total_score', 'ies_mean', 'rse_total', 'ssrq_total', 'fruit_veg_portion',
       'ceq_q1', 'ceq_q2', 'ceq_q3', 'ceq_q4', 'ceq_q5', 'ceq_q6']

# Simple behavior
behavior_simple = ['login_day_0', 'login_day_1', 'login_day_2', 'login_day_3', 'login_day_4', 'login_day_5', 'login_day_6']

# Simple behavior
behavior = ['days_for_s1', 'days_for_s2', 'char_answers_len', 'answers_total', 'sec_spend_week1', 'days_login_w1',
            'msg_coach_len', 'msg_group_len', 'msg_coach_num', 'msg_group_num', 'num_diary_entries',
            'char_diary_entries', 'char_diary_letter']

# Extended behavior
behavior_all = ['SessionsCom_day1', 'SessionsCom_day2', 'SessionsCom_day3', 'SessionsCom_day4', 'SessionsCom_day5','SessionsCom_day6', 'SessionsCom_day7',
                'sec_spend_day1', 'sec_spend_day2', 'sec_spend_day3', 'sec_spend_day4', 'sec_spend_day5','sec_spend_day6', 'sec_spend_day7',
                'sec_spentFrSaSun', 'sec_spentMoTu', 'sec_spentWedThur',
                'sec_spentday', 'sec_spentevening', 'sec_spentmorning',
                'login_day_0', 'login_day_1', 'login_day_2', 'login_day_3', 'login_day_4', 'login_day_5', 'login_day_6',
                'pagerequest_day1', 'pagerequest_day2', 'pagerequest_day3', 'pagerequest_day4', 'pagerequest_day5', 'pagerequest_day6','pagerequest_day7',
                'char_answers_day1', 'char_answers_day2', 'char_answers_day3', 'char_answers_day4', 'char_answers_day5','char_answers_day6', 'char_answers_day7',
                'answers_total_day1', 'answers_total_day2', 'answers_total_day3', 'answers_total_day4', 'answers_total_day5', 'answers_total_day6', 'answers_total_day7',
                'msg_coach_lenchar_day0', 'msg_coach_lenchar_day1', 'msg_coach_lenchar_day2', 'msg_coach_lenchar_day3', 'msg_coach_lenchar_day4', 'msg_coach_lenchar_day5', 'msg_coach_lenchar_day6',
                'msg_group_lenchar_day0', 'msg_group_lenchar_day1', 'msg_group_lenchar_day2', 'msg_group_lenchar_day3', 'msg_group_lenchar_day4', 'msg_group_lenchar_day5', 'msg_group_lenchar_day6',
                'msg_coach_lencount_day0', 'msg_coach_lencount_day1', 'msg_coach_lencount_day2', 'msg_coach_lencount_day3', 'msg_coach_lencount_day4', 'msg_coach_lencount_day5', 'msg_coach_lencount_day6',
                'msg_group_lencount_day0', 'msg_group_lencount_day1', 'msg_group_lencount_day2','msg_group_lencount_day3', 'msg_group_lencount_day4', 'msg_group_lencount_day5', 'msg_group_lencount_day6',
                'num_diary_day1', 'num_diary_day2', 'num_diary_day3', 'num_diary_day4', 'num_diary_day5','num_diary_day6', 'num_diary_day7',
                'char_diary_week1',
                'Abfuehrmittel_Entwaesserungsmitt_mean','Abfuehrmittel_Entwaesserungsmitt_min','Abfuehrmittel_Entwaesserungsmitt_max',
                'Appetitzuegler_mean','Appetitzuegler_min', 'Appetitzuegler_max',
                'Erbrechen_mean', 'Erbrechen_min', 'Erbrechen_max',
                'Essanfaelle_mean', 'Essanfaelle_min', 'Essanfaelle_max',
                'Mahlzeiten_insgesamt_mean', 'Mahlzeiten_insgesamt_min', 'Mahlzeiten_insgesamt_max',
                'Mahlzeiten_nicht_satt_mean', 'Mahlzeiten_nicht_satt_min', 'Mahlzeiten_nicht_satt_max',
                'Nahrungsmittel_verzichtet_mean', 'Nahrungsmittel_verzichtet_min', 'Nahrungsmittel_verzichtet_max',
                'anzahl_fastfood_fertiggerichte_mean', 'anzahl_fastfood_fertiggerichte_min','anzahl_fastfood_fertiggerichte_max',
                'anzahl_mahlzeiten_mean','anzahl_mahlzeiten_min', 'anzahl_mahlzeiten_max',
                'anzahl_mahlzeiten_ausgewogen_mean', 'anzahl_mahlzeiten_ausgewogen_min','anzahl_mahlzeiten_ausgewogen_max',
                'anzahl_mahlzeiten_nichthungrig_mean', 'anzahl_mahlzeiten_nichthungrig_min','anzahl_mahlzeiten_nichthungrig_max',
                'anzahl_mahlzeiten_uebersaettigt_mean', 'anzahl_mahlzeiten_uebersaettigt_min','anzahl_mahlzeiten_uebersaettigt_max',
                'glaeser_gezuckerte_getraenke_mean', 'glaeser_gezuckerte_getraenke_min','glaeser_gezuckerte_getraenke_max',
                'portionen_obst_gemuese_mean', 'portionen_obst_gemuese_min', 'portionen_obst_gemuese_max',
                'zwischen_mahlzeiten_gegessen_mean', 'zwischen_mahlzeiten_gegessen_min','zwischen_mahlzeiten_gegessen_max']

run_names = {'Simple Questionnaire': 'wcs_only',
             'Extended Questionnaire': 'baseline_extended',
             'User Behavior Simple': 'behavior_simple',
             'User Behavior Extended': 'behavior_extended',
             'User Behavior Selected': 'behavior_selected',
             'All Features': 'all_features'}
