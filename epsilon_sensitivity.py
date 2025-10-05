import numpy as np
import matplotlib.pyplot as plt

eps = np.array([0.50, 0.75, 1.00, 1.25, 1.50])
logA = np.array([9.193, 9.089, 8.997, 8.926, 8.893])

# 线性拟合（logA = a * eps + b）
coeffs = np.polyfit(eps, logA, deg=1)
fit = np.poly1d(coeffs)
slope, intercept = coeffs

plt.figure(figsize=(6,4))
plt.scatter(eps, logA, color='tab:blue', label='Runs')
plt.plot(eps, fit(eps), color='tab:orange', label=f'Linear fit: logA = {slope:.3f} ε + {intercept:.3f}')
plt.axvline(1.0, color='gray', ls='--', alpha=0.6, label='Baseline ε=1.0')
plt.xlabel(r'$\epsilon$ scale factor')
plt.ylabel(r'Final NLTE $\log A(\mathrm{O})$')
plt.title('Sensitivity of Derived Abundance to $\epsilon$ scaling')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('epsilon_sensitivity.png', dpi=300)
plt.close()
print(f'Slope ≈ {slope:.3f} dex per unit ε (≈ {(logA[0]-logA[2]):+.3f} dex when ε halved)')
print('Plot saved as epsilon_sensitivity.png')