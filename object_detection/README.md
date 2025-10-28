## 🧾 Step-by-Step Instructions


### 🧩 Step 1: Setup Python Environment
``` python3 -m venv bdd```

Activate environment ``` source bdd/bin/activate```

Install dependencies from requirements.txt
```pip install -r requirements.txt```


### 🧩 Step 2: Download Model checkpoint
Download the dataset from the link below and place it inside the `analysis/` directory.

**[Download checkpoint](https://drive.google.com/file/d/1Ochq_lJf0IQJlG_lX0SskijT-RVZKuRz/view?usp=sharing)**

place the checkpoint in ```/BDD100k-object-Detection/object_detection/bdd```


### 🧩 Step 3: Download Dataset
Download the dataset from the link below:

**[Download Dataset](https://drive.google.com/file/d/1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT/view?usp=sharing)**

### 🧩 Step 4: Modify Config file
Update key values in train_im_sets, train_label_sets in 
``` config/bdd.yaml```
