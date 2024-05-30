import torch
import matplotlib.pyplot as plt
import tiktoken
from model import TransformerLanguageModel

# Assuming you have a saved model
#model = torch.load('model-ckpt.pt')

# Or if you are using torch.load with a specific model class
model = TransformerLanguageModel()
model.load_state_dict(torch.load('model-ckpt_1.pt'))
state_dict = model.state_dict()

model0 = TransformerLanguageModel()
model0.load_state_dict(torch.load('model-ckpt_0.pt'))
state_dict0 = model0.state_dict()

# Print parameter names and sizes
for name, param in state_dict.items():
    print(name, param.size())

# Access a specific parameter by name
embedding_weight = state_dict['token_embedding_lookup_table.weight']
#print("token_embedding_lookup_table.weight:", embedding_weight[:1])

embedding_weight0 = state_dict0['token_embedding_lookup_table.weight']
#print("token_embedding_lookup_table.weight_0:", embedding_weight0[:1])


differences = torch.norm(embedding_weight - embedding_weight0, dim=1)
print("difference is ", differences.shape)
most_changed_token_index = torch.argmax(differences).item()
print("Most changed token index:", most_changed_token_index)
#print("Initial embedding:", embedding_weight0[most_changed_token_index])
#print("Final embedding:", embedding_weight[most_changed_token_index])
print("Change in embedding:", differences[most_changed_token_index])

encoding = tiktoken.get_encoding("cl100k_base")
print('---------------')
print(encoding.decode([most_changed_token_index]))
print('---------------')

top_k = 5
sorted_indices = torch.argsort(differences, descending=True)[:top_k]
for idx in sorted_indices:
    token_index = idx.item()
    print(f"{encoding.decode([idx])},          {token_index},    {differences[token_index].item():.2f}")

# Step 2: Convert tensor to numpy array for plotting
tensor_np = embedding_weight.numpy()
tensor_np0 = embedding_weight0.numpy()





#handle the other model pair
model_b1 = TransformerLanguageModel()
model_b1.load_state_dict(torch.load('model-ckpt_b1.pt'))
state_dict_b1 = model_b1.state_dict()
model_b0 = TransformerLanguageModel()
model_b0.load_state_dict(torch.load('model-ckpt_b0.pt'))
state_dict_b0 = model_b0.state_dict()
embedding_weight_b1 = state_dict_b1['token_embedding_lookup_table.weight']
embedding_weight_b0 = state_dict_b0['token_embedding_lookup_table.weight']
tensor_np_b1 = embedding_weight_b1.numpy()
tensor_np_b0 = embedding_weight_b0.numpy()






index=5845
training_diff=tensor_np[index, :] - tensor_np0[index, :]
training_diff_b=tensor_np_b1[index, :] - tensor_np_b0[index, :]
diff_diff = training_diff_b - training_diff

# Step 3: Plot each column as a separate curve
plt.figure(figsize=(10, 6))
#for i in range(index,index+1): #tensor_np.shape[0]):
for i in [index]:
    print("plot index = ", i)
    plt.plot(tensor_np0[i, :], label=f'Row{i}_0')
    plt.plot(training_diff, label=f'Change{i}')
    plt.plot(diff_diff, label='diff_diff')


# Step 4: Add titles and labels
plt.title(encoding.decode([index]))
plt.xlabel('Dim')
plt.ylabel('Value')
plt.legend()
plt.show()
