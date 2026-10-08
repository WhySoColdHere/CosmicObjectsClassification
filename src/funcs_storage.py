class FuncsStorage:
    @staticmethod
    def get_dataset_structure(dataset_columns):
        learn_columns = [col for col in dataset_columns if col != 'filepath' and col.endswith('fraction')]
        separator = '-ukidss_'
        column_prefix = lambda x: x[:x.find(separator)]
        questions_answers = dict((q, []) for q in map(column_prefix, learn_columns))
        for col in learn_columns:
            questions_answers[column_prefix(col)].append(col)

        return questions_answers