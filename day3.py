df['quality'] = df['quality'].astype(str)  
plt.figure(figsize=(10, 8))
sns.violinplot(x="quality", y="alcohol", data=df, palette={
'3': 'lightcoral', '5': 'lightblue', '8': 'lightgreen', '8': 'gold', '5': 'lightskyblue', '4': 'lightpink'}, alpha=0.7)
plt.title('Violin Plot for Quality and Alcohol')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()