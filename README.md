# Extractive Question Answering Using Recipe Data

Modifying RecipeQA Dataset from https://hucvl.github.io/recipeqa/

Original source datasets from RecipeQA have been downloaded and can be accessed from our shared folder https://drive.google.com/drive/folders/1oWazhCfs2S2XAKmf5RcbRkkKFTqpsfMq?usp=share_link

Fine tuning Generative Question Answering task learnt from Hugging Face course: https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt

## Requirements

Install dependencies used in the project using the command
```
pip install -r requirements.txt
```

## Files
- `making_dataset.py`: Contains the `QADataset` class for downloading and converting the question answering dataset from original dataset.
- `preprocessing.py`: Contains the `Preprocessing` class responsible for preprocessing the input data.
- `model.py`: Contains the `Model` class to perform fine-tuning on different pre-trained models
- `hpo.py`: Contains the `HPO` class to perform hyper-parameter optimization.

## Classes

### `QADataset`
- Download data from Google Drive and convert data to desired format for Extractive Question Answering
- Methods:
    - `filter_data_from_original()`: create two dataframes, textual and visual from textual_cloze and visual_coherence
    - `combine_steps()`: combine all step titles in the recipe set
    - `combine_textual_visual_df()`: merge and clean textual and visual dataframe created
    - `generate_questions()`: create questions for the dataset
    - `generate_full_instruction()`: create full context/instruction for each recipe
    - `generate_answer_and_index()`: create spanned answer and answer start index
    - `make_final_data()`: make the final data (apply for train, test, and validation set)
    
### `Preprocessing`
- Preprocess data to get ready for training processes
- Methods:
    - `make_dataset()`: convert dataframes into DatasetDict structure to easier access train and val set 
    - `tokenizer()`: call instance of tokenizer by given pretrained model checkpoint
    - `preprocess_training()`: Preprocesses the training instances by tokenizing the questions and context, and generating start and end positions for the answer spans.
Returns the preprocessed inputs.
    - `preprocess_validation()`: Preprocesses the validation instances by tokenizing the questions and context, 
    and modifying the offset mapping to handle tokenization differences between training and validation examples.
Returns the preprocessed inputs.
    - `preprocessed_train()`: Apply `preprocess_training()` function to the whole training set and return results
    - `preprocessed_val()`: Apply `preprocess_validation()` function to the whole validation set and return results
    - `train_dataloader()`: Creates a PyTorch DataLoader for the preprocessed training dataset. Returns the training DataLoader.
    - `eval_dataloader()`: Creates a PyTorch DataLoader for the preprocessed validation dataset. Returns the validation DataLoader.
    

### `Model`
- Inherits from `Preprocessing` class.
- Implements the fine-tuning process for the question answering model.
- Methods:
  - `call_model()`: Loads the pre-trained question answering model.
  - `prepare_fine_tuning()`: Prepares the model, optimizer, and data loaders for fine-tuning.
  - `train()`: Performs the training loop.
  - `evaluate()`: Performs the evaluation on the validation set.
  - `fine_tune()`: Executes the fine-tuning process.

### `HPO`
- Inherits from `Model` class.
- Implements hyper-parameter optimization for the fine-tuned model and calculate test set performace.
- Methods:
  - `hyperparameter_opimization()`: Optimizes the hyper-parameters.

## Functions

- `download_data_source()`: Download source train, test, validation data in .json format to the current working directory (can be found in `model.py`)
- `compute_metrics()`: Computes evaluation metrics for the question answering model. (can be found in `model.py`)
- `calculate_test()`: Calculate test set performance on the optimized model. (can be found in `hpo.py`)

## Usage

- Notice that results from `hpo.py` can be obtained separated from `model.py`. However, the project flow is run from `model.py` to fine-tune models first, 
then `hpo.py` to optimize hyper-parameters 

### model.py

The code is executed inside the `if __name__ == '__main__':` block. When the script is run, it performs fine-tuning using different pre-trained models and prints the results.

To run the code:

```
python model.py
```

Notice that inside this module, there is variable `checkpoint_list` containing all the pre-trained models that all team members have fine-tuned on the dataset.
However, there are some concerns:
- In order to save time, shouldn't run all models. All fine-tuned models can be reached here at our HF hubs:
    - Dhruv's: https://huggingface.co/slushi7
    - Gianni's: https://huggingface.co/Data255FinalProj
    - Saumya's: https://huggingface.co/saumyasinha0510
    - Tam's: https://huggingface.co/tamhuynh27
- All the models were initially trained and ran on Google Colab. 
However, running on your own machine with different configurations can cause some errors such as: 
`TypeError: Operation 'neg_out_mps() does not support input type 'int64' in MPS backend.` (Got this error when fine-tuning or calling `Deberta` model on macOS. 
This is due to model architecture and operating system, not to source code.)
- Suggest to start with a small model to test the code as left default `checkpoint_list = ['distilbert-base-uncased']`
- In order to obtain the same result from the report and presentation, comment out the distilbert checkpoint, and uncomment the following `checkpoint_list = ['microsoft/deberta-base']`
### hpo.py

The code is executed inside `if __name__ == '__main__':` block. When the script is run, it performs hyperparameter optimization on the best fine-tuned model `slushi7/deberta-base-recipeQA`.
And calculate performance of the test set on the optimized model from checkpoint `Data255FinalProj/deberta-saumya-tune` 

```
python hpo.py
```
