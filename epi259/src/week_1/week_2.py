import numpy as np

N=30420
n_vacc = 14134
n_placebo = 14073
prob_vaccine = 1/2
prob_placebo = 1/2

#%%
placebo_covid = 185/14073
placebo_covid_py = 13 * 1000  # py

placebo_rate_infection = 56.5 * 1000  # py

# 1. Determine the average length of time that participants
# in the placebo group were followed during the study.

numb_events = 0
event_py = 0
incidence_rate = numb_events * event_py
