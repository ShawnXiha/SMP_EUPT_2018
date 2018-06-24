from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from keras import backend as K
import numpy as np


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))


class F1Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.name = ''
    def set_name(self, name):
        self.name = name
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=256, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)

            print("\n ROC-AUC -score - epoch: %d - score: %.6f \n" % (
                epoch + 1, score))
            if score > self.max_score:
                print(
                    "*** New High Score (previous: %.6f) \n" % self.max_score)
                self.model.save_weights(f"~/total_data/{self.name}best_weights_with_score_{score}.h5")
                self.max_score = score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                old_lr = float(K.get_value(self.model.optimizer.lr))
                new_lr = old_lr * 0.99
                if self.not_better_count > 5 or new_lr < 1e-5:
                    print("Epoch %05d: early stopping, high score = %.6f" % (
                        epoch, self.max_score))
                    self.model.stop_training = True
                else:
                    K.set_value(self.model.optimizer.lr, new_lr)




