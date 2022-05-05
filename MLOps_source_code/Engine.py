import pandas as pd
import matplotlib.pyplot as plt
import pylab
import scipy
from matplotlib import pyplot
from MLPipeline.Gaussian_Stationary import Gaussian_Stationary
from MLPipeline.Gaussian_Trend import Gaussian_Trend

# importing the data
raw_csv_data = pd.read_excel("Input/CallCenterData.xlsx")

# check point of data
df_comp = raw_csv_data.copy()

# date to numeric
df_comp["timestamp"] = df_comp["month"].apply(lambda x : x.timestamp())

# Setting date index
# taken as a date time field
df_comp.set_index("month", inplace=True)

# seeting the frequency as monthly
df_comp = df_comp.asfreq('M')

df_comp.Healthcare.plot(figsize=(20,5), title="Healthcare")
plt.savefig("Output/"+"dataplot_healthcare.png")


# Check for normality

# Density Plots
df_comp["Healthcare"].plot(kind='kde', figsize=(20, 10))
pyplot.savefig("Output/"+"Densityplot.png")

# The QQ plot
scipy.stats.probplot(df_comp["Healthcare"], plot=pylab)
plt.title("QQ plot for Healthcare")
pylab.savefig("Output/"+"QQPLot.png")


# Gaussian Processes
data_df = df_comp[["timestamp", "Healthcare"]]

# Gaussian Trend
Gaussian_Trend(data_df)


# Difference
df_comp["delta_1_Healthcare"] = df_comp.Healthcare.diff(1)

df_comp.delta_1_Healthcare.plot(figsize=(20,5))

# Checking the normality again with Density Plots
df_comp["delta_1_Healthcare"].plot(kind='kde', figsize=(20, 10))
pyplot.savefig("Output/"+"difference.png")

data_df_res = df_comp[["timestamp", "delta_1_Healthcare"]]

# Gaussian Stationary
Gaussian_Stationary(df_comp, data_df_res)
