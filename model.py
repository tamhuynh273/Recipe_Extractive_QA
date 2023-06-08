# import preprocessing and making_dataset modules
from preprocessing import Preprocessing
from making_dataset import QADataset

import gdown
import os
import numpy as np
from transformers import AutoModelForQuestionAnswering, get_scheduler
from tqdm.auto import tqdm
import evaluate
import collections
import torch
from torch.optim import AdamW
from accelerate import Accelerator
import warnings
warnings.filterwarnings("ignore")


class Model(Preprocessing):
    def __init__(self, train_df, val_df, checkpoint, device, accelerator):
        super().__init__(train_df, val_df, checkpoint)
        self.device = device
        self.accelerator = accelerator

    def call_model(self):
        return AutoModelForQuestionAnswering.from_pretrained(self.checkpoint).to(self.device)

    def prepare_fine_tuning(self, lr, num_train_epochs):
        model = self.call_model()

        accelerator = self.accelerator
        optimizer = AdamW(model.parameters(), lr=lr)  # function parameter
        train_dataloader = self.train_dataloader()
        eval_dataloader = self.eval_dataloader()

        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        return model, optimizer, train_dataloader, eval_dataloader, lr_scheduler

    # training loop
    def train(self, model, optimizer, train_dataloader, lr_scheduler, progress_bar):
        model.train()
        accelerator = self.accelerator
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # evaluating loop
    def evaluate(self, model, eval_dataloader):
        model.eval()
        accelerator = self.accelerator
        start_logits = []
        end_logits = []

        validation_dataset = self.preprocessed_val()

        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        squad_metric, bleu_metric = compute_metrics(
            start_logits, end_logits, validation_dataset, self.dataset['val'])

        return squad_metric, bleu_metric

    def fine_tune(self, num_epoch=3, lr=2e-5):
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = self.prepare_fine_tuning(
            lr=lr, num_train_epochs=num_epoch)

        progress_bar = tqdm(range(num_epoch*len(train_dataloader)))

        print("\n*** Generating Results... ***\n")
        for epoch in range(num_epoch):
            self.train(model, optimizer, train_dataloader, lr_scheduler, progress_bar)
            squad_metric, bleu_metric = self.evaluate(model, eval_dataloader)

            print("Epoch", epoch)
            print("SQuAD:", squad_metric)
            print("BLEU score:", bleu_metric['bleu'])


def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 10
    max_answer_length = 10
    predicted_answers = []

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    data = {'predictions': [], 'references': []}

    prediction_list = []
    reference_list = []

    for prediction, reference in zip(predicted_answers, theoretical_answers):
        data['predictions'].append({'id': str(prediction['id']), 'prediction_text': prediction['prediction_text']})
        data['references'].append({'id': str(reference['id']),
                                   'answers': [{'text': ans, 'answer_start': start} for ans, start in
                                               zip(reference['answers']['text'],
                                                   reference['answers']['answer_start'])]})

        prediction_list.append(prediction['prediction_text'])
        reference_list += reference['answers']['text']

    squad_metric = evaluate.load("squad").compute(predictions=data['predictions'], references=data['references'])
    bleu_metric = evaluate.load("bleu").compute(predictions=prediction_list, references=reference_list)

    return squad_metric, bleu_metric

def download_data_source():
    data_file_list = ['train', 'test', 'val']
    url_list = ['https://drive.google.com/file/d/1zThNvSNVf5KTN2mRYRnwhDHHHdQuIjak/view?usp=share_link',
                'https://drive.google.com/file/d/1rGbVWVybwpl8kqwxJxOVhLSBceU4u4SL/view?usp=share_link',
                'https://drive.google.com/file/d/1iFazzaNp_YJPsQbuO99X8w0fqSQVQZGk/view?usp=share_link']

    for file, url in zip(data_file_list, url_list):
        output_path = os.path.join(os.getcwd(), f"{file}.json")

        gdown.download(url, output_path, quiet=False, fuzzy=True)


if __name__ == '__main__':
    if not os.path.exists('train.json') or not os.path.exists('val.json') or not os.path.exists('test.json'):
        print('File not exist, start downloading...')
        download_data_source()

    train = QADataset(datafile='train.json')
    validation = QADataset(datafile='val.json')
    test = QADataset(datafile='test.json')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    accelerator = Accelerator()

    """
    checkpoint_list include all the pre-trained models that we have applied fine-tuning on
    can modify to reduce the number of models for faster computational time for testing the codes
    best performing model is 'microsoft/deberta-base'
    """
    # checkpoint_list = ["microsoft/deberta-base", "google/electra-small-generator", "google/electra-large-generator",
    #                    "albert-base-v2", "distilbert-base-uncased", "nghuyong/ernie-2.0-base-en", "xlm-roberta-base",
    #                    "facebook/bart-base", "Google/bigbird-roberta-base", "roberta-base",
    #                    "squeezebert/squeezebert-uncased", "bert-base"]

    # recommend following checkpoint for the fastest way to test code
    checkpoint_list = ['distilbert-base-uncased']

    # uncomment following line, and comment line 212 to run the best performing model
    # checkpoint_list = ['microsoft/deberta-base']

    for checkpoint in checkpoint_list:
        model = Model(train_df=train.data, val_df=validation.data, checkpoint=checkpoint,
                      device=device, accelerator=accelerator)
        print("\n********************************")
        print("Fine tuning with `" + checkpoint + "` pretrained model:\n")
        model.fine_tune()
        print("********************************\n")





