# Text Generation with GPT 2
## Fine-Tunning GPT-2 for Text Generation

This project demonstrates how to train a model to generate coherent and contextually relevent text based on a given prompt. GPT-2 is fine tuned on custom dataset to create text that mimics style and structure of training data. 

#### 1. Set Up Your Virtual Environment:

First, create a virtual environment to isolate your project dependencies. Run the following command:

    python -m venv .venv

Activate the virtual environment:

- On Windows:

        .venv\Scripts\activate

- On macOS/Linux:
        
        source .venv/bin/activate

Install the required dependencies:

    pip install -r requirements.txt

#### 2. Prepare the Dataset:

- Create a Data Input Folder: Create a folder named `data_input/` at the root of your project.

- Place Your Data: Place your dataset in the `data_input/` folder.

#### 3. Train the Model:

Run the `train.py` script located in the `src/` directory to fine-tune the GPT-2 model on your custom dataset:
    
    python src/train.py

This will train the model and save it in a folder named pretrained/, which you need to create if it doesn’t exist.

#### 4. Generate Text

After training, you can generate text using the fine-tuned model by running the `main.py ` script:

    python main.py
