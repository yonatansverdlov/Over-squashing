
# Over-squash

---

## Description

Over-squash is a tool for solving the problem of information bottlenecks in graph neural networks (GNNs) or implementing effective mechanisms to mitigate over-squashing issues. This project uses Python and deep learning frameworks, including PyTorch and PyTorch Geometric. It aims to enhance the ability of GNNs to handle long-range dependencies without suffering from information loss or compression, thereby improving their performance in tasks requiring deep relational information.

---

## Installation For Over-squashing and transductive learning

To set up the project environment and install all necessary dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yonatansverdlov/Over-squashing.git
   ```

2. Navigate into the project directory:
   ```bash
   cd Over-squashing
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

We present three types of experiments: Over-squashing experiments, Transductive learning, MolHIV and LRGB.
## Graph classification tasks
First run
   ```bash
   cd bottleneck/script
   ```
Select a `data_type` from the following nine options: **Protein,MUTAG**.
Run:
   ```bash
   python train.py --dataset_name data_type --repeat 10 -min_radius 3 --max_radius 4
   ```
## Over-squashing experiments
First run
   ```bash
   cd bottleneck/script
   ```
Choose data_type, one of the four options: Ring, Tree, CrossRing, CliquePath. 
Then, for Tree, choose a radius between 2 and 8, and for others, between 2 and 15.
For non-Tree
Run
   ```bash
   python train.py --dataset_name data_type --min_radius 2 --max_radius 16
   ```
For Tree run
   ```bash
   python train.py --dataset_name data_type --min_radius 2 --max_radius 9
   ```
---
## Transductive Learning
First run
   ```bash
   cd bottleneck/script
   ```
Select a `data_type` from the following nine options: **Cora, Cite, Pubm, Cham, Squi, Actor, Corn, Texas, Wisc**.
Run:
   ```bash
   python train.py --dataset_name data_type --repeat 10 
   ```
## LRGB & MolHIV
1. Create a new Conda environment and activate it:
   ```bash
   cd Over-squashing
   conda create --name lrgb -c conda-forge python=3.10
   conda activate lrgb
   ```
2. Install the necessary dependencies from the `lrgb_requirements.txt` file:
   ```bash
   pip install -r lrgb_requirements.txt
   ```
## License

This project is licensed under the MIT License.

---

## Contact

For any questions or feedback, reach out to me at `yonatans@campus.technion.ac.il`.
