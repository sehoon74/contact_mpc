import pickle
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42

pickles = ['cem.p', 'mpc.p', 'cemmpc.p']

bar_width = 0.8/len(pickles)
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
for i, p in enumerate(pickles):
    d = pickle.load(open(p, 'rb'))
    bar_positions = [pos + i * bar_width for pos in range(len(d.keys()))]  # Adjust positions
    if i==0:
        plt.xticks(bar_positions, list(d.keys()))
    plt.bar(bar_positions, d.values(), label=p.split('.')[0], width=bar_width, align='center')


    
# Add labels and title
#plt.xlabel('Data Points')
plt.ylabel('Cost')

# Add legend
plt.legend()

# Show the plot
plt.grid(visible=True, axis='y')
plt.tight_layout()
plt.show()
