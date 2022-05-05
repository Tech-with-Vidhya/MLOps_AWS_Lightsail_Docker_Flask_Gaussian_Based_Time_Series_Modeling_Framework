


# Gaussian Process
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor

class Gaussian_Stationary:
    
    def __init__(self, df_comp, data_df_res):
        
        
        self.split_data(data_df_res, df_comp)

        self.set_Kernel()

        self.get_model()

        self.train()

    def train(self):
        """
        Training GP1 model
        :return:
        """
        self.gp1.fit(self.x_train_res_1, self.y_train_res_1)
        # Generate predictions.
        y_pred, y_std = self.gp1.predict(self.x_train_res_1, return_std=True)
        self.df_train_res['y_pred'] = y_pred
        self.df_train_res['y_std'] = y_std
        self.df_train_res['y_pred_lwr'] = self.df_train_res['y_pred'] - 2 * self.df_train_res['y_std']
        self.df_train_res['y_pred_upr'] = self.df_train_res['y_pred'] + 2 * self.df_train_res['y_std']
        plt.figure(figsize=(20, 5))
        plt.plot(self.df_train_res["y_pred"], color='red')
        plt.plot(self.df_train_res["delta_1_Healthcare"])
        plt.savefig("Output/"+"pred_delta_train.png")
        # Generate predictions.
        y_pred, y_std = self.gp1.predict(self.x_test_res_1, return_std=True)
        self.df_test_res['y_pred'] = y_pred
        self.df_test_res['y_std'] = y_std
        self.df_test_res['y_pred_lwr'] = self.df_test_res['y_pred'] - 2 * self.df_test_res['y_std']
        self.df_test_res['y_pred_upr'] = self.df_test_res['y_pred'] + 2 * self.df_test_res['y_std']
        plt.figure(figsize=(20, 5))
        plt.plot(self.df_test_res["y_pred"])
        plt.plot(self.df_test_res["delta_1_Healthcare"], color='red')
        plt.savefig("Output/"+"pred_stationary.png")

    def get_model(self):
        """
        Setting the hyperparameters
        :return:
        """
        self.gp1 = GaussianProcessRegressor(
            kernel=self.kernel_1,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=0.004
        )

    def set_Kernel(self):
        """
        Setting up the kernel
        :return:
        """
        k0 = WhiteKernel(noise_level=0.3 ** 2, noise_level_bounds=(0.1 ** 2, 0.5 ** 2))
        k1 = ConstantKernel(constant_value=2) * \
             ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45))
        self.kernel_1 = k0 + k1

    def split_data(self, data_df_res, df_comp):
        """
        Spliting the data into train and test
        :param data_df_res:
        :param df_comp:
        :return:
        """
        # train set split
        test_size = 12
        X = df_comp["timestamp"]
        y = df_comp["delta_1_Healthcare"]
        x_train_res = X[:-test_size]
        y_train_res = y[:-test_size]
        x_test_res = X[-test_size:]
        y_test_res = y[-test_size:]
        self.df_train_res = data_df_res[:-test_size][1:]
        self.df_test_res = data_df_res[-test_size:][1:]
        plt.figure(figsize=(20, 5))
        plt.title('train and test sets', size=20)
        plt.plot(y_train_res, label='Training set')
        plt.plot(y_test_res, label='Test set', color='orange')
        plt.legend()
        plt.savefig("Output/"+"split.png")
        self.x_train_res_1 = x_train_res.values.reshape(-1, 1)[1:]
        self.y_train_res_1 = y_train_res.values.reshape(-1, 1)[1:]
        self.x_test_res_1 = x_test_res.values.reshape(-1, 1)[1:]
        self.y_test_res_1 = y_test_res.values.reshape(-1, 1)[1:]



    





