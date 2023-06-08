import torch
from accelerate import Accelerator
from transformers import pipeline
import evaluate
import os

from making_dataset import QADataset
from model import Model


class HPO(Model):
    def __init__(self, train_df, val_df, checkpoint, device, accelerator):
        super().__init__(train_df, val_df, checkpoint, device, accelerator)

    def hyperparameter_opimization(self, lr, num_epoch):
        self.fine_tune(lr=lr, num_epoch=num_epoch)


def calculate_test(test_df):
    # Load the optimized fine-tuned model
    optimized_model = 'Data255FinalProj/deberta-saumya-tune'
    model = pipeline('question-answering', model=optimized_model)

    # Evaluate the model on the test set
    predicted_answers = []
    ground_truth_answers = []
    data = {'predictions': [], 'references': []}

    for i in range(len(test_df)):
        context = test_df['context'][i]
        question = test_df['question'][i]
        ground_truth_answer = test_df['answers'][i]['text']
        id = test_df['id'][i]

        # Generate the predicted answer from the model

        result = model({'context': context, 'question': question})
        predicted_answer = result['answer']

        predicted_answers.append(predicted_answer)
        ground_truth_answers += ground_truth_answer

        data['predictions'].append({'id': str(id), 'prediction_text': predicted_answer})
        data['references'].append({'id': str(id), 'answers': test_df['answers'][i]})

    squad_metric = evaluate.load("squad").compute(predictions=data['predictions'], references=data['references'])
    bleu_metric = evaluate.load("bleu").compute(predictions=predicted_answers, references=ground_truth_answers)

    return squad_metric, bleu_metric


if __name__ == '__main__':
    if not os.path.exists('train.json') or not os.path.exists('val.json') or not os.path.exists('test.json'):
        print('File not exist, start downloading...')
        Model.download_data_source()

    train = QADataset(datafile='train.json')
    validation = QADataset(datafile='val.json')
    test = QADataset(datafile='test.json')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    accelerator = Accelerator()

    """
    best_fine_tuned_checkpoint has already been used as fine-tuning the modified recipe data
    on QA task and been pushed to HF hub by one of our teammate
    """

    best_fine_tuned_checkpoint = "slushi7/deberta-base-recipeQA"
    # best_fine_tuned_checkpoint = "tamhuynh27/bert-finetuned-squad"
    hpo = HPO(train_df=train.data, val_df=validation.data, checkpoint=best_fine_tuned_checkpoint,\
              device=device, accelerator=accelerator)

    print("\n********************************")
    print("Optimizing hyper-parameter with best fine-tuned `" + best_fine_tuned_checkpoint + "` model\n")
    hpo.hyperparameter_opimization(lr=2e-6, num_epoch=5)
    print("********************************\n")

    print("\n********************************")
    print("Test set performance with optimized model\n")
    squad_metric, bleu_metric = calculate_test(test.data)
    print("SQuAD:", squad_metric)
    print("BLEU score:", bleu_metric['bleu'])
    print("\n********************************")


