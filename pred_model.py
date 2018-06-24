import numpy as np
my_dict = {0: '人类作者', 1: '机器作者', 2: '机器翻译', 3: '自动摘要'}


def apply_model_to_all_df(model, model_weight, data, store_path):
    model.load_weights(model_weight)
    y_pred = model.predict(data, batch_size=256,
                                verbose=1)
    y_p = np.argmax(y_pred, 1)
    y = np.vectorize(my_dict.get)(y_p)
    np.save(store_path, y)


if __name__ == '__main__':
    from models.bigru_with_cnn import build_model
    from config import *
    model = build_model()
    model_weight = "./models/best_weights.h5"
    data = np.load('./inputs/data.npz')['x_train']
    store_path = "./inputs/train_y_pred.npy"
    apply_model_to_all_df(model, model_weight, data, store_path)

