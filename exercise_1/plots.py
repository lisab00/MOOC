import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

main_dir = Path(r'C:\Users\lihel\Documents\MOOC\exercise_1')
data = pd.read_csv(main_dir / "rc_Q_V.txt", sep=',')

### Model 1
# sample densities
samples = pd.read_csv(main_dir / "outputs" / "model_1" / "samples" / "sam_var9_n06.txt", header=None)
true_params = [-0.03,1.9]

rows, cols = 1, 2
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for ax, i in zip(axes.flat, np.arange(2)):
    sns.kdeplot(samples[i], cumulative=False, ax=ax, warn_singular=False, label="Sample density")
    ax.axvline(true_params[i], color="#ff7f0e", linestyle="--", linewidth=2, label="True value")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.legend()
    ax.set_title(f"Sample density of parameter {'a' if i == 0 else 'b'}", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.6)

    
plt.savefig(main_dir / "outputs" / "model_1" / "sam_var9_n06.png")    
plt.show()

# Rating curves 
a = -0.04 #true prms
b = 1.87

ah = 0.34 #estimated prms
bh = 1.75

plt.scatter(np.log(data['H']), np.log(data['Q']), label="Logarithmic data")
plt.plot(np.log(data['H']), a + b* np.log(data['H']), color = "orange",label="Rating curve by formula")
plt.title("Model 1: rating curve against data (linear form)")
plt.xlabel("Water height (cm)")
plt.ylabel("Flow (l/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
plt.close()

plt.scatter(data['H'], data['Q'], label="Data")
plt.plot(np.linspace(20,59,100), np.exp(a + b * np.log(np.linspace(20,59,100))), color = "orange",label="Analytical rating curve")
plt.plot(np.linspace(20,59,100), np.exp(ah + bh * np.log(np.linspace(20,59,100))), color = "green",label="Estimated rating curve")
plt.title("Exponential model: rating curve against data")
plt.xlabel("Water height (cm)")
plt.ylabel("Flow (l/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
plt.close()



### Model 2
# sample densities
samples = pd.read_csv(main_dir / "outputs" / "model_2" / "samples" / "sam_var238_n1.txt", header=None)
true_params = [-0.78, -18.5, 1.18]

rows, cols = 1, 3
fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

for ax, i in zip(axes.flat, np.arange(3)):
    sns.kdeplot(samples[i], cumulative=False, ax=ax, warn_singular=False, label="Sample density")
    ax.axvline(true_params[i], color="#ff7f0e", linestyle="--", linewidth=2, label="True value")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.legend()
    ax.set_title(f"Sample density of parameter {'a' if i == 0 else 'b' if i==1 else 'c'}", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.6)

    
plt.savefig(main_dir / "outputs" / "model_2" / "sam_var238_n1.png")    
plt.show()


# Rating curve
a = -0.000235
b = 0.0041
c = 0.621

ah = 0.07
bh = -0.14
ch = 0.68

plt.scatter(data['H'], data['Q'], label="Data")
plt.plot(np.linspace(20,59,100), a + b * np.linspace(20,59,100) + c* np.linspace(20,59,100)**2, color = "orange",label="Analytical rating curve")
plt.plot(np.linspace(20,59,100), ah + bh * np.linspace(20,59,100) + ch* np.linspace(20,59,100)**2, color = "green",label="Estimated rating curve")
plt.title("Polynomial model: rating curve against data")
plt.xlabel("Water height (cm)")
plt.ylabel("Flow (l/s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
plt.close()


