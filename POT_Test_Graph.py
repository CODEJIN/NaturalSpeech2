import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

with open('./POT_Test/Exp3015.POT_Test.txt', 'r') as f:
    frameworks, fixed_lengths, devices, input_lengths, times = zip(*[
        line.strip().split('\t')
        for line in f.readlines()[1:]
        ])
    
input_lengths = [int(x) for x in input_lengths]
times = [float(x) for x in times]

data = {
    'Framework': frameworks,
    'Fixed_Length': fixed_lengths,
    'Device': devices,
    'Input_Length': input_lengths,
    'Time': times,
    }
df = pd.DataFrame(data)

frigre, axs = plt.subplots(1, 2, figsize= (20, 10))
for device_index, device in enumerate(['CPU', 'GPU']):
    df_device = df[df['Device'] == device]
    ax = sns.barplot(
        x= 'Framework',
        y= 'Time',
        hue= 'Fixed_Length',
        data= df_device,
        errorbar= ('ci', 95),
        ax= axs[device_index]
        )
    axs[device_index].set_title(f'Device: {device}')
    for index in ax.containers:
        ax.bar_label(index,)

plt.tight_layout()
plt.savefig('./POT_Test/Exp3015.POT_Test_Bar.png')