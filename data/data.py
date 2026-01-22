import pandas as pd

df = pd.read_csv("data/household_power_consumption.csv", sep=";", na_values="?")

df["time"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
df = df.set_index("time")

df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

df["energy_kWh_min"] = df["Global_active_power"] / 60 

energy_hourly = df["energy_kWh_min"].resample("1h").sum()

out = energy_hourly.reset_index()
out.columns = ["time", "energy_kWh"]

out["energy_kWh"] = out["energy_kWh"].round(3)

out.to_csv("data/dataset_energy_hourly.csv", index=False)
