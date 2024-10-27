
# Over-squash

---

## Description

Over-squash is a tool for solving the problem of information bottlenecks in graph neural networks (GNNs) or implementing effective mechanisms to mitigate over-squashing issues. This project is built using Python and deep learning frameworks, including PyTorch and PyTorch Geometric. It aims to enhance the ability of GNNs to handle long-range dependencies without suffering from information loss or compression, thereby improving their performance in tasks requiring deep relational information.

---

## Installation

To set up the project environment and install all necessary dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Over-squash.git
   ```

2. Navigate into the project directory:
   ```bash
   cd Over-squash/bottleneck
   ```

3. Create a new Conda environment and activate it:
   ```bash
   conda create --name oversquash -c conda-forge python=3.11
   conda activate oversquash
   ```

4. Install the necessary dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To run the project, follow these steps:

1. Activate the Conda environment:
   ```bash
   conda activate oversquash
   ```
   
## Over-squashing experiments
Choose data_type, one of the four options: Ring, Tree, CrossRing, CliqueRing. 
Then, for Tree, choose a radius between 2 and 8, and for others, between 2 and 15.
And run
   ```bash
   python train.py --dataset_name data_type --radius radius.
   ```
---
If all radios are needed, please run
   ```bash
   python train.py --dataset_name data --all True
   ```
## Trunsductive Learning

## LRGB
## License

This project is licensed under the MIT License.

---

## Contact

For any questions or feedback, reach out to me at `yonatans@campus.technion.ac.il`.
