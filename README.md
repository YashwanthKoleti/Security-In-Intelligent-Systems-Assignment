# Security-In-Intelligent-Systems-Assignment

In this assignment we trained a baseline model, data poisoning, adversarial attacks, and robust model training.

## **Assignment Structure**

```
├── archive/                       -> MNIST dataset
├── Assignment 1 - Secure AI Systems.pdf   # Assignment.pdf
├── baseline_model.py              -> Baseline MNIST model training
├── baseline_model.h5              -> Saved baseline model
├── confusion_matrix_baseline.png  -> Baseline model confusion matrix
├── Attack.ipynb                   -> Data poisoning, FGSM/PGD attacks, robust training
├── Report.pdf                     -> Final assignment report
└── README.md
```

* **baseline_model.py** — trains the baseline model.
* **Attack.ipynb** — performs:

  * data poisoning
  * FGSM & PGD adversarial sample generation
  * evaluation on clean/poisoned/adversarial data
  * robust adversarial training
* **confusion_matrix_baseline.png** — baseline model confusion matrix on clean data.
* **Report.pdf**

## **How to Run**

### Train baseline model

```bash
python baseline_model.py
```

### Run data poisoning,adversarial attacks, adversial model training

Open the notebook:

```bash
Attack.ipynb
```

## **Dataset**

MNIST files are stored in the **archive/** directory.

Team:
- @https://github.com/YashwanthKoleti
- @https://github.com/harsh241082
