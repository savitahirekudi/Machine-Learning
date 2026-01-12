import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "name": ["Apple", "Banana", "Cherry"],
    "value": [10, 25, 15]
})

plt.bar(df["name"], df["value"])
plt.xlabel("Fruit")
plt.ylabel("Quantity")
plt.title("Bar Chart Example")
plt.show()