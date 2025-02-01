# Over-Squash

## Description

Over-squashing is a phenomenon in Graph Neural Networks (GNNs) where information bottlenecks hinder effective message passing over long-range dependencies. This project implements mechanisms to mitigate over-squashing, improving GNN performance on tasks requiring deep relational reasoning. It is built using Python and deep learning frameworks, including **PyTorch** and **PyTorch Geometric**.

## Installation for Over-Squashing and Transductive Learning

To set up the project environment and install all necessary dependencies, follow these steps:

```bash
git clone https://github.com/yonatansverdlov/Over-squashing.git
cd Over-squashing
conda env create -f dependencies.yaml
conda activate oversquash
```

## Usage

This project includes three types of experiments: **Over-Squashing Experiments, Transductive Learning, Graphh Classification, and MolHIV & LRGB.**

### Over-Squashing Experiments
Navigate to the bottleneck directory:
```bash
cd bottleneck
```
Choose a `data_type` from the following options: **Ring, Tree, CrossRing, CliquePath.**  
For `Tree`, choose a radius between **2 and 8**, for others, between **2 and 15**.

For `Tree` experiments:
```bash
python train.py --dataset_name data_type --min_radius 2 --max_radius 9
```
For `Ring`, `CrossRing`, or `CliquePath`:
```bash
python train.py --dataset_name data_type --min_radius 2 --max_radius 16
```

### Transductive Learning
Navigate to the bottleneck directory:
```bash
cd bottleneck
```
Choose a `data_type` from the following options: **Cora, Cite, Pubm, Cham, Squi, Actor, Corn, Texas, Wisc.**  
Set `repeat` to a value between **1 and 10** to determine the number of different seeds:
```bash
python train.py --dataset_name data_type --repeat repeat
```

### Graph Classification (Mutag & Protein)
For `MUTAG` and `Protein` datasets, specify the number of different seeds (`repeat`) and run:
```bash
python train.py --dataset_name data_type --repeat repeat --min_radius 3 --max_radius 4
```
For LRGB and MolHIV:
```bash
cd Over-squashing
conda create --name lrgb -c conda-forge python=3.10
conda activate lrgb
pip install -r lrgb_requirements.txt
```

## License

This project is licensed under the MIT License.

## Contact

For any questions or feedback, reach out to me at `yonatans@campus.technion.ac.il`.

