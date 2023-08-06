from sklearn.metrics import average_precision_score

class Evaluator:
    def __init__(self, metric: str):
        if metric not in {'ap', 'mae'}:
            raise NotImplementedError()
        self.metric = metric

    def eval(self, y: dict):
        if self.metric == 'ap':
            return self._eval_ap(y['y_true'], y['y_pred'])
        elif self.metric == 'mae':
            return self._eval_mae(y['y_true'], y['y_pred'])

    def _eval_ap(self, y_true, y_pred):
        """
            compute Average Precision (AP) averaged across tasks
        """

        ap_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if (y_true[:,i] == 1).sum() > 0 and (y_true[:,i] == 0).sum() > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return {'ap': sum(ap_list)/len(ap_list)}
    
    def _eval_mae(self, y_true, y_pred):
        """
            compute Mean Absolute Error (MAE) averaged across tasks
        """
        mae_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            mae = (y_true[is_labeled, i] - y_pred[is_labeled, i]).abs().mean()
            mae_list.append(mae)
        
        return {'mae': sum(mae_list)/len(mae_list)}