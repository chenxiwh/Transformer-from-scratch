import torch
import matplotlib.pyplot as plt

# Step 1: Create a sample tensor (shape: rows x columns)
tensor = torch.randn(100, 5)  # 100 rows, 5 columns

# Step 2: Convert tensor to numpy array for plotting
tensor_np = tensor.numpy()

# Step 3: Plot each column as a separate curve
plt.figure(figsize=(10, 6))
for i in range(tensor_np.shape[1]):
    plt.plot(tensor_np[:, i], label=f'Column {i + 1}')

# Step 4: Add titles and labels
plt.title('Columns of Tensor as Curves')
plt.xlabel('Row Index')
plt.ylabel('Value')
plt.legend()
plt.show()
