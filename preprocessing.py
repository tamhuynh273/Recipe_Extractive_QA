from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import DatasetDict, Dataset

class Preprocessing():
    def __init__(self, train_df, val_df, checkpoint, max_length=512, stride=64):
        self.train_df = train_df
        self.val_df = val_df
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.stride = stride
        self.dataset = self.make_dataset()

    def make_dataset(self):
        train = Dataset.from_pandas(self.train_df)
        val = Dataset.from_pandas(self.val_df)
        full_dataset = DatasetDict({'train': train, 'val': val})

        return full_dataset

    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.checkpoint)

    def preprocess_training(self, examples):
        questions = [q.strip() for q in examples["question"]]
        tokenizer = self.tokenizer()
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs

    def preprocess_validation(self, examples):
        questions = [q.strip() for q in examples["question"]]
        tokenizer = self.tokenizer()
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids

        return inputs

    def preprocessed_train(self):
        return self.dataset['train'].map(self.preprocess_training, \
                                         batched=True, remove_columns=self.dataset['train'].column_names)

    def preprocessed_val(self):
        return self.dataset['val'].map(self.preprocess_validation, \
                                       batched=True, remove_columns=self.dataset['val'].column_names)

    def train_dataloader(self, batch_size=8):
        train_dataset = self.preprocessed_train()
        train_dataset.set_format("torch")

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=batch_size
        )
        return train_dataloader

    def eval_dataloader(self, batch_size=8):
        val_ = self.preprocessed_val()
        val_dataset = val_.remove_columns(['example_id', 'offset_mapping'])
        val_dataset.set_format('torch')

        val_dataloader = DataLoader(
            val_dataset,
            collate_fn=default_data_collator,
            batch_size=batch_size
        )
        return val_dataloader


if __name__ == '__main__':
    pass
