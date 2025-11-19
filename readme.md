# IMDB Movie Review Classification Project Description

This is a sentiment classification project based on the IMDB movie review dataset, supporting the following features:

- Basic model training
- Loading pre-trained models for interactive testing
- Ensemble inference by combining multiple models (using the FedAvg strategy)

---

## Project Setup

1. **Check GPU and CUDA Installation**
- Press `Win + R`, type `cmd` to open the Command Prompt
- Run the command:
```bash
  nvidia-smi
```
Make sure the GPU driver and CUDA version are displayed correctly.

2. **Environment Setup**
- Use **Anaconda** to create a virtual environment
- Choose **Python 3.10.10** as the base interpreter

3. **Install the GPU Version of PyTorch**
```bash
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

4. **Check that the PyTorch Version is 2.8.0+cu129**
```bash
  which cuda
  python -c "import torch; print(torch.__version__)"
```
5. **Install Other Dependencies**
- Install other dependencies such as scikit-learn and numpy according to PyCharm's prompts

---
## 项目结构

1. **Main Files**

        project/
        ├── Tips---
        │ ├── bin_check/                # Model framework compatibility check scripts
        │ ├── log.py                    # Development logging module
        │ └── torch_version_control.py  # Torch version control command-line tool
        ├── .gitignore                  # Git ignore file configuration
        ├── config.py                   # Hyperparameter configuration (learning rate, batch size, etc.)
        ├── data_loader/                # Data loading and cleaning logic
        ├── dataset.py                  # Custom Dataset class and data preprocessing
        ├── interactive_inference.py    # Interactive inference interface (users can input movie reviews for testing)
        ├── merge_and_evaluate.py       # Model ensemble module (includes data fetching, splitting, parameter allocation algorithms, fully independent)
        ├── model.py                    # BERT classification model definition
        ├── train.py                    # Main entry for model training
        ├── visualize.py                # Training process visualization (e.g., loss/accuracy curves)
        └── output_of_Fedavg/           # Console output logs for FedAvg model aggregation


**2. Generated Files**

After running train.py, the following are generated:

        best_bert_model(1).bin
        best_bert_model(2).bin
        ...
        best_bert_model(n).bin

Each file corresponds to the best model weights saved from a single training session.

After running merge_and_evaluate.py, the following are generated:
        
        fedavg_merged_bert_model.bin

Results from the model fusion

---
## References and Third-Party Libraries
1. **References**
    
   McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.

2. **Third-Party Libraries**
    
    pandas;re;matplotlib;seaborn;nltk;wordcloud;PyTorch;TensorFlow