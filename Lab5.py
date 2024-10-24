import pandas as pd
import numpy as np
import time

# TASK 1

start1 = time.time()
pdfore = pd.read_csv("vic_elec_forecast.csv")
end1 = time.time()
print("Time taken by loading pdfore: {:.2f} seconds".format(end1 - start1))
#print(pdfore)

start2 = time.time()
npfore = np.loadtxt("vic_elec_forecast.csv", delimiter=',', skiprows=1, usecols=(1, 2, 4, 5, 6, 7), dtype=None)
#print(npfore)
end2 = time.time()
print("Time taken by loading npfore: {:.2f} seconds".format(end2 - start2))
#print(npfore)

pd_total_time = 0
np_total_time = 0
np_power_total_time = 0
num_repeats = 100

demand_np = npfore[:,0]
npfore_3 = npfore[:,3]
npfore_4 = npfore[:,4]
npfore_5 = npfore[:,5]

for i in range(num_repeats):
    start_time = time.time()

    # Compute MAE, RMSE, and MAPE for Demand_forecast1 using pandas
    demand = pdfore['Demand']
    #pd_mae = np.mean(np.abs(demand - pdfore['Demand_forecast1']))
    #pd_rmse = np.sqrt(np.mean((demand - pdfore['Demand_forecast1']) ** 2))
    #pd_mape = np.mean(np.abs((demand - pdfore['Demand_forecast1']) / demand)) * 100
    pd_mae = pdfore["Demand_forecast1"].sub(demand).abs().mean()
    pd_rmse = ((pdfore["Demand_forecast1"] - demand) ** 2).mean() ** 0.5
    pd_mape = ((pdfore["Demand_forecast1"] - demand).abs() / demand).mean() * 100

    # Compute MAE, RMSE, and MAPE for Demand_forecast2 using pandas
    pd_mae2 = pdfore["Demand_forecast2"].sub(demand).abs().mean()
    pd_rmse2 = ((pdfore["Demand_forecast2"] - demand) ** 2).mean() ** 0.5
    pd_mape2 = ((pdfore["Demand_forecast2"] - demand).abs() / demand).mean() * 100

    # Compute MAE, RMSE, and MAPE for Demand_forecast3 using pandas
    pd_mae3 = pdfore["Demand_forecast3"].sub(demand).abs().mean()
    pd_rmse3 = ((pdfore["Demand_forecast3"] - demand) ** 2).mean() ** 0.5
    pd_mape3 = ((pdfore["Demand_forecast3"] - demand).abs() / demand).mean() * 100

    end_time = time.time()
    pd_total_time += end_time - start_time

    start_time = time.time()

    # Compute MAE, RMSE, and MAPE for Demand_forecast1 using numpy and (array)**2
    #print(demand_np)
    np_mae = np.mean(np.abs(demand_np - npfore_3))
    np_rmse = np.sqrt(np.mean((npfore_3 - demand_np) ** 2))
    np_mape = np.mean(np.abs((demand_np - npfore_3) / demand_np)) * 100

    # Demand_forecast2
    np_mae2 = np.mean(np.abs(demand_np - npfore_4))
    np_rmse2 = np.sqrt(np.mean((npfore_4 - demand_np) ** 2))
    np_mape2 = np.mean(np.abs((demand_np - npfore_4) / demand_np)) * 100

    # Demand_forecast3
    np_mae3 = np.mean(np.abs(demand_np - npfore_5))
    np_rmse3 = np.sqrt(np.mean((npfore_5 - demand_np) ** 2))
    np_mape3 = np.mean(np.abs((demand_np - npfore_5) / demand_np)) * 100

    end_time = time.time()
    np_total_time += end_time - start_time

    start_time = time.time()

    # Compute MAE, RMSE, and MAPE for Demand_forecast1 using numpy and power()
    np_mae = np.mean(np.abs(demand_np - npfore_3))
    np_rmse = np.sqrt(np.mean(np.power((npfore_3 - demand_np), 2)))
    np_mape = np.mean(np.abs((demand_np - npfore_3) / demand_np)) * 100

    # Demand_forecast2
    np_mae2 = np.mean(np.abs(demand_np - npfore_4))
    np_rmse2 = np.sqrt(np.mean(np.power((npfore_4 - demand_np), 2)))
    np_mape2 = np.mean(np.abs((demand_np - npfore_4) / demand_np)) * 100

    # Demand_forecast3
    np_mae3 = np.mean(np.abs(demand_np - npfore_5))
    np_rmse3 = np.sqrt(np.mean(np.power((npfore_5 - demand_np), 2)))
    np_mape3 = np.mean(np.abs((demand_np - npfore_5) / demand_np)) * 100

    end_time = time.time()
    np_power_total_time += end_time - start_time


print(f"Pandas demand_forecast1: MAE={pd_mae:.2f}, RMSE={pd_rmse:.2f}, MAPE={pd_mape:.2f}%")
print(f"Pandas demand_forecast2: MAE={pd_mae2:.2f}, RMSE={pd_rmse2:.2f}, MAPE={pd_mape2:.2f}%")
print(f"Pandas demand_forecast3: MAE={pd_mae3:.2f}, RMSE={pd_rmse3:.2f}, MAPE={pd_mape3:.2f}%")

print(f"Numpy demand_forecast1: MAE={np_mae:.2f}, RMSE={np_rmse:.2f}, MAPE={np_mape:.2f}%")
print(f"Numpy demand_forecast2: MAE={np_mae2:.2f}, RMSE={np_rmse2:.2f}, MAPE={np_mape2:.2f}%")
print(f"Numpy demand_forecast2: MAE={np_mae3:.2f}, RMSE={np_rmse3:.2f}, MAPE={np_mape3:.2f}%")


print(f"Pandas total time: {pd_total_time:.4f}s")
print(f"Numpy total time: {np_total_time:.4f}s")
print(f"Numpy + power total time: {np_power_total_time:.4f}s")

# Numpy is much faster than Pandas.

'''
The power() function is faster than array**2, for a few reasons:

The np.power() function can broadcast the exponent to an array of the same shape as the base, 
which can save memory and computation time compared to using the ** operator.

The np.power() function is more numerically stable than the ** operator, 
particularly for large exponents or bases that are close to zero or infinity. 
This can help prevent issues like overflow or underflow, which can slow down computation.

The np.power() function is implemented in optimized C code, 
which is designed to take advantage of the specific hardware 
and software environment on which it is running. 
This can make it faster than the ** operator, which is implemented in Python 
and may not be as well-optimized for the specific task at hand.
'''

# Create a new column that is the average of the three forecast columns
Demand_forecast_avg = pdfore[["Demand_forecast1", "Demand_forecast2", "Demand_forecast3"]].mean(axis=1)
pd_mae_avg = Demand_forecast_avg.sub(demand).abs().mean()
pd_rmse_avg = ((Demand_forecast_avg - demand) ** 2).mean() ** 0.5
pd_mape_avg = ((Demand_forecast_avg - demand).abs() / demand).mean() * 100

print(f"Pandas demand_forecast1: MAE={pd_mae:.2f}, RMSE={pd_rmse:.2f}, MAPE={pd_mape:.2f}%")
print(f"Pandas demand_forecast2: MAE={pd_mae2:.2f}, RMSE={pd_rmse2:.2f}, MAPE={pd_mape2:.2f}%")
print(f"Pandas demand_forecast3: MAE={pd_mae3:.2f}, RMSE={pd_rmse3:.2f}, MAPE={pd_mape3:.2f}%")
print(f"Pandas demand_forecast_avg: MAE={pd_mae_avg:.2f}, RMSE={pd_rmse_avg:.2f}, MAPE={pd_mape_avg:.2f}%")
# Errors are much lower!

# TASK 2
'''
prop = np.loadtxt("vic_elec_prob.csv", delimiter=' ', dtype=None)
#print(prop)

# compute the lower and upper bounds of the 50%, 90%, and 98% prediction intervals
bounds = np.percentile(prop, [25, 75, 5, 95, 1, 99], axis=1)
lower_50, upper_50, lower_90, upper_90, lower_98, upper_98 = bounds

# compute the coverage of the 50%, 90%, and 98% prediction intervals
coverage_50 = np.mean((demand >= lower_50) & (demand <= upper_50))
coverage_90 = np.mean((demand >= lower_90) & (demand <= upper_90))
coverage_98 = np.mean((demand >= lower_98) & (demand <= upper_98))

print("Coverage of 50% prediction interval: {:.2f}%".format(coverage_50*100))
print("Coverage of 90% prediction interval: {:.2f}%".format(coverage_90*100))
print("Coverage of 98% prediction interval: {:.2f}%".format(coverage_98*100))
'''