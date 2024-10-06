import plant_scheduler.nutrients as ps
import numpy as np
import pandas as pd

DATA_PATH_AVG = '/home/mach/Projects/PlantNutrition/data/input/nutrients_avg.xlsx'

# np.random.seed = 45

c_fertilizer1 = np.abs(np.random.sample(11))*10
c_fertilizer2 = np.abs(np.random.sample(11))*3
c_fertilizer3 = np.abs(np.random.sample(11))*0.1
    
df_results = ps.calculate_fertilization_schedule(data_path=DATA_PATH_AVG,
                                               time_intervall_days=2,
                                               c_fertilizer1=c_fertilizer1,
                                               c_fertilizer2=c_fertilizer2,
                                               c_fertilizer3=c_fertilizer3)
    


ps.plot_excess_stacked_interactive(df_results)
# print(df_results.head(n=20))
  
with pd.ExcelWriter("/home/mach/Projects/PlantNutrition/data/results/results.xlsx") as writer:
    df_results.to_excel(writer)  

