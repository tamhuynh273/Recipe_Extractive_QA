import pandas as pd
import json
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 1000)


class QADataset():
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = self.make_final_data(datafile)

    # create two dataframes, textual and visual from textual_cloze and visual_coherence
    def filter_data_from_original(self):
        f = open('{dataset}'.format(dataset=self.datafile), 'r')

        read_data = json.loads(f.read())
        textual_json = [x for x in read_data['data'] if x['task'] == 'textual_cloze']
        visual_json = [x for x in read_data['data'] if x['task'] == 'visual_coherence']

        textual_df = pd.DataFrame(textual_json)
        textual_keep_col = ['recipe_id', 'context', 'choice_list', 'answer', 'question']
        textual_df = textual_df[textual_keep_col]

        visual_df = pd.DataFrame(visual_json)
        visual_keep_col = ['recipe_id', 'context']
        visual_df = visual_df[visual_keep_col]

        return textual_df, visual_df

    # combine all steps into list
    def combine_steps(self, row):
        steps = []
        num_steps = len(row.context)
        for step in range(num_steps):
            steps.append(row.context[step]['title'])

        return steps

    # combine textual and visual dataframe
    def combine_textual_visual_df(self, datafile):
        textual_df, visual_df = self.filter_data_from_original()
        visual_df['all_steps'] = visual_df.apply(lambda row: self.combine_steps(row), axis=1)
        combined_data = pd.merge(textual_df, visual_df, how='inner', on=['recipe_id'])
        combined_data.rename(columns={'context_x': 'context'}, inplace=True)
        combined_data = combined_data[['recipe_id', 'context', 'choice_list', 'answer', 'question', 'all_steps']]
        combined_data.question = combined_data.question.apply(
            lambda x: [i.replace('@placeholder', '_') if i == '@placeholder' else i for i in x])

        return combined_data

    # generate questions
    def generate_questions(self, row):
        create_question = ""
        given_question = row['question']
        all_steps = row['all_steps']
        target_index = given_question.index('_')  # index of question in the given list
        if target_index == 0:
            # check if it's the first step in full steps
            temp_idx = all_steps.index(given_question[target_index + 1])
            if temp_idx == 1:
                create_question = "What is the first step?"
            else:
                create_question = "What is the step after " + all_steps[temp_idx - 2] + " ?"
        elif target_index == 3:
            if all_steps.index(given_question[target_index - 1]) == (
                    len(all_steps) - 2):  # check if the question step is the last step
                create_question = "What is the last step?"
            else:
                create_question = "What is the step after " + given_question[target_index - 1] + " ?"
        else:
            create_question = "What is the step after " + given_question[target_index - 1] + " ?"

        return create_question

    # generate context
    def generate_full_instruction(self, row):
        full_instruction = ""
        context = row["context"]
        steps = row["all_steps"]

        for step in range(len(steps)):
            if step == 0:
                full_instruction += "The first step is " + str(steps[step]) + ": " + context[step]['body'] + ". "
            elif step == (len(steps) - 1):
                full_instruction += "The last step is " + str(steps[step]) + ": " + context[step]['body'] + ". "
            else:
                full_instruction += "After the previous step is " + str(steps[step]) + ": " + context[step][
                    'body'] + ". "
        # clean instruction
        full_instruction = re.sub('\s+', ' ', full_instruction).strip()
        full_instruction = re.sub(r'\n', ' ', full_instruction)
        return full_instruction[0:-1]

    # generate answers
    def generate_answer_and_index(self, row):
        actual_answer = {}
        answer = row["choice_list"][row.answer]
        actual_answer["text"] = [answer]
        full_instruction = row.full_instruction

        actual_answer["answer_start"] = [full_instruction.find(answer)]
        return actual_answer

    def make_final_data(self, datafile):
        combine_data = self.combine_textual_visual_df(datafile)
        combine_data['full_instruction'] = combine_data.apply(lambda row: self.generate_full_instruction(row), axis=1)
        combine_data['new_question'] = combine_data.apply(lambda row: self.generate_questions(row), axis=1)
        combine_data['actual_answer'] = combine_data.apply(lambda row: self.generate_answer_and_index(row), axis=1)
        dup_check = combine_data[['recipe_id', 'full_instruction', 'new_question']]
        combine_data = combine_data[dup_check.duplicated() == False].reset_index(drop=True)
        final_data = combine_data[['recipe_id', 'full_instruction', 'new_question', 'actual_answer']].reset_index()
        final_data.rename(
            columns={'index': 'id', 'recipe_id': 'title', 'full_instruction': 'context', 'new_question': 'question',
                     'actual_answer': 'answers'}, inplace=True)
        return final_data


if __name__ == '__main__':
    pass



