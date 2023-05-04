from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
labeling_function = lambda row: 0

class BaselineNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    @step
    def start(self):

        # Step-level dependencies are loaded within a Step, instead of loading them 
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = np.where(_has_review_df['rating'] > 3, 1, 0)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({'label': labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})
        self.train_df, self.validation_df = train_test_split(_df, test_size=self.split_size)
        print(f'num of rows in train set: {self.train_df.shape[0]}')
        print(f'num of rows in validation set: {self.validation_df.shape[0]}')

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.metrics import accuracy_score, roc_auc_score
        import numpy as np
        np.random.seed(42)
        
        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        self.baseline_predictions = np.random.randint(0, 2, size=self.validation_df.shape[0])
        self.base_acc = accuracy_score(self.validation_df.label, self.baseline_predictions)
        self.base_rocauc = roc_auc_score(self.validation_df.label, self.baseline_predictions)
        self.next(self.end)
        
    @card(type='corise')
    @step
    def end(self):

        print(f"Baseline Accuracy: {self.base_acc:.3f}")
        print(f"Baseline AUC: {self.base_rocauc:.3f}")

        current.card.append(Markdown("# Womens Clothing Review Results"))

        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        current.card.append(Markdown("## Examples of False Positives"))
        f = self.baseline_predictions == 1
        f &= self.validation_df['label'] == 0
        self.false_pos = self.validation_df.where(f).dropna(how='all')
        current.card.append(Table.from_dataframe(self.false_pos[['review']].sample(5)))
        
        current.card.append(Markdown("## Examples of False Negatives"))
        f = self.baseline_predictions == 0
        f &= self.validation_df['label'] == 1
        self.false_neg = self.validation_df.where(f).dropna(how='all')
        current.card.append(Table.from_dataframe(self.false_neg[['review']].sample(5)))

if __name__ == '__main__':
    BaselineNLPFlow()
