import pandas as pd
import numpy as np
data = {
'School ID': [101, 102, 103, np.nan, 105, 106, 107, 108],
'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
'Address': ['123 Main St', '456 Oak Ave', '789 Pine Ln', '101 Elm St', np.nan, '222 Maple Rd', '444 Cedar Blvd', '555 Birch Dr'],
'City': ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', np.nan, 'Pune', 'Jaipur'],
'Subject': ['Math', 'English', 'Science', 'Math', 'History', 'Math', 'Science', 'English'],
'Marks': [85, 92, 78, 89, np.nan, 95, 80, 88],
'Rank': [2, 1, 4, 3, 8, 1, 5, 3],
'Grade': ['B', 'A', 'C', 'B', 'D', 'A', 'C', 'B']
}
df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)