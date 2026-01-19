# 📁Distributed Water Resources Optimization

A Python implementation of a distributed consensus algorithm for optimizing agricultural water allocation across multiple regions.

## 💎 Quick Start

### 🔧 Installation
Ensure you have Python 3.7+ installed. Then, install the required dependencies:
```bash
pip install numpy pandas matplotlib
```
Basic Usage
Clone this repository.
The main algorithm is contained in water_allocation.py.
Run the optimization with the following command, which uses the default parameters from the study:
```bash
import numpy as np
from water_allocation import hanshu

# Set the parameters as used in the research
np.random.seed(18)
hanshu(kstep=600, alpha=0.01, gamma=5, theta=1, beta=10, gamma_trans=1, gamma_irricost=5, N=23, ratio=0.8, max_de=93.78)
```
Input Data
```bash
The algorithm is configured with built-in data for 23 regions, including:

Cultivated land area
Crop water demands
Transportation distances
Crop coefficients (Ky)
```
Output
```bash
The script will print key results to the console, including:

Water allocation before and after optimization
Total yield improvements
Fairness analysis (Coefficient of Variation)
```
Citation
If you use this code in your research, please cite:

[Your Paper Title]. [Authors]. [Year]. [Journal/Conference Name].

License
This project is licensed under the GPL 3.0 License.
