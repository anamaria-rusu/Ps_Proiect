import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(n_patches, d_model):
    pe = np.zeros((n_patches, d_model))
    position = np.arange(0, n_patches).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


N = 256      
D = 14        
pos_highlight = 64 

pe_matrix = get_positional_encoding(N, D)

# --- Randare Grafic ---

fig, ax = plt.subplots(figsize=(10, 6))

offset = 5 # offset intre unde
t_axis = np.arange(N)

for i in range(D):
    
    x_values = pe_matrix[:, i] + i * offset
    ax.plot(x_values, t_axis, linewidth=0.8, alpha=0.8)
    
    
    val = pe_matrix[pos_highlight, i]
    ax.scatter(val + i * offset, pos_highlight, s=15, zorder=5)
    
    
    ax.text(val + i * offset + 0.2, pos_highlight, f'{val:.4f}', fontsize=9, verticalalignment='center',
            bbox=dict(alpha=0.5, edgecolor='none'))


rect_width = D * offset
rect = plt.Rectangle((-1, pos_highlight - 5), rect_width, 10, facecolor='none', linewidth=1.5, alpha=0.6)
ax.add_patch(rect)


ax.set_ylabel('pos', fontsize=14)
ax.set_xlabel('$D_{model}$', fontsize=14)
ax.set_yticks(np.arange(0, N + 1, 32))
ax.set_xticks([i * offset for i in range(D)])
ax.set_xticklabels([str(i) for i in range(D)])



for spine in ['top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)

plt.title(f"Positional Encoding Matrix\n$N={N}, D_{{model}}={D}$", pad=20)
plt.tight_layout()
plt.savefig("TRF/plots/positional_encoding_matrix.pdf", format='pdf')
plt.show()