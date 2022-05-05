

# Gaussian Process

from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error


class Gaussian_Trend:

    def __init__(self, df_comp):
        sns.set_style(
            style='darkgrid',
            rc={'axes.facecolor': '.9', 'grid.color': '.8'}
        )
        sns.set_palette(palette='deep')
        self.sns_c = sns.color_palette(palette='deep')

        self.set_Kernel()
        self.get_model()

        self.split_data(df_comp)

        self.train()


    def train(self):
        """
        Function to train the Gaussian Process
        :return:
        """
        self.gp2.fit(self.x_train, self.y_train)
        # Generate predictions.
        y_pred, y_std = self.gp2.predict(self.x_train, return_std=True)
        self.df_train['y_pred'] = y_pred
        self.df_train['y_std'] = y_std
        self.df_train['y_pred_lwr'] = self.df_train['y_pred'] - 2 * self.df_train['y_std']
        self.df_train['y_pred_upr'] = self.df_train['y_pred'] + 2 * self.df_train['y_std']
        # Generate predictions.
        y_pred_test, y_std_test = self.gp2.predict(self.x_test, return_std=True)
        self.df_test['y_pred'] = y_pred_test
        self.df_test['y_std'] = y_std_test
        self.df_test['y_pred_lwr'] = self.df_test['y_pred'] - 2 * self.df_test['y_std']
        self.df_test['y_pred_upr'] = self.df_test['y_pred'] + 2 * self.df_test['y_std']
        plt.figure(figsize=(20, 5))
        plt.plot(self.df_train["y_pred"])
        plt.plot(self.df_train["Healthcare"], color='red')
        plt.savefig("Output/"+"pred_train.png")
        print(f'R2 Score Train = {self.gp2.score(X=self.x_train, y=self.y_train): 0.3f}')
        print(f'R2 Score Test = {self.gp2.score(X=self.x_test, y=self.y_test): 0.3f}')
        print(f'MAE Train = {mean_absolute_error(y_true=self.y_train, y_pred=self.gp2.predict(self.x_train)): 0.3f}')
        print(f'MAE Test = {mean_absolute_error(y_true=self.y_test, y_pred=self.gp2.predict(self.x_test)): 0.3f}')
        errors = self.gp2.predict(self.x_train) - self.y_train
        errors = errors.flatten()
        errors_mean = errors.mean()
        errors_std = errors.std()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.regplot(x=self.y_train.flatten(), y=self.gp2.predict(self.x_train).flatten(), ax=ax[0])
        sns.distplot(a=errors, ax=ax[1])
        ax[1].axvline(x=errors_mean, color=self.sns_c[3], linestyle='--', label=f'$\mu$')
        ax[1].axvline(x=errors_mean + 2 * errors_std, color=self.sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
        ax[1].axvline(x=errors_mean - 2 * errors_std, color=self.sns_c[4], linestyle='--')
        ax[1].axvline(x=errors_mean, color=self.sns_c[3], linestyle='--')
        ax[1].legend()
        ax[0].set(title='Model 2 - Train vs Predictions (Train Set)', xlabel='y_train', ylabel='y_pred');
        ax[1].set(title='Model 2  - Errors', xlabel='error', ylabel=None);
        errors = self.gp2.predict(self.x_test) - self.y_test
        errors = errors.flatten()
        errors_mean = errors.mean()
        errors_std = errors.std()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.regplot(x=self.y_test.flatten(), y=self.gp2.predict(self.x_test).flatten(), ax=ax[0])
        sns.distplot(a=errors, ax=ax[1])
        ax[1].axvline(x=errors_mean, color=self.sns_c[3], linestyle='--', label=f'$\mu$')
        ax[1].axvline(x=errors_mean + 2 * errors_std, color=self.sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
        ax[1].axvline(x=errors_mean - 2 * errors_std, color=self.sns_c[4], linestyle='--')
        ax[1].axvline(x=errors_mean, color=self.sns_c[3], linestyle='--')
        ax[1].legend()
        ax[0].set(title='Model 2 - Test vs Predictions (Test Set)', xlabel='y_test', ylabel='y_pred');
        ax[1].set(title='Model 2  - Errors', xlabel='error', ylabel=None)
        plt.savefig("Output/"+"predictions.png")

    def split_data(self, df_comp):
        """
        Spliting the data into train test
        :param df_comp:
        :return:
        """
        # train set split
        test_size = 22
        X = df_comp["timestamp"]
        y = df_comp["Healthcare"]
        x_train_t = X[:-test_size]
        y_train_t = y[:-test_size]
        x_test_t = X[-test_size:]
        y_test_t = y[-test_size:]
        self.df_train = df_comp[:-test_size]
        self.df_test = df_comp[-test_size:]
        plt.figure(figsize=(20, 5))
        plt.title('train and test sets', size=20)
        plt.plot(y_train_t, label='Training set')
        plt.plot(y_test_t, label='Test set', color='orange')
        plt.legend();
        plt.savefig("Output/"+"split_data.png")
        self.x_train = x_train_t.values.reshape(-1, 1)
        self.y_train = y_train_t.values.reshape(-1, 1)
        self.x_test = x_test_t.values.reshape(-1, 1)
        self.y_test = y_test_t.values.reshape(-1, 1)

    def get_model(self):
        """
        Setting the model parameters
        :return:
        """
        self.gp2 = GaussianProcessRegressor(
            kernel=self.kernel_4,
            n_restarts_optimizer=10,
            normalize_y=True,
            alpha=0.0
        )

    def set_Kernel(self):
        """
        Setting up the kernel
        :return:
        """
        k0 = WhiteKernel(noise_level=0.3 ** 2, noise_level_bounds=(0.1 ** 2, 0.5 ** 2))
        k1 = ConstantKernel(constant_value=2) * \
             ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45))
        k2 = ConstantKernel(constant_value=100, constant_value_bounds=(1, 500)) * \
             RationalQuadratic(length_scale=500, length_scale_bounds=(1, 1e4), alpha=50.0, alpha_bounds=(1, 1e3))
        k3 = ConstantKernel(constant_value=1) * \
             ExpSineSquared(length_scale=1.0, periodicity=12, periodicity_bounds=(10, 15))
        self.kernel_4 = k0 + k1 + k2 + k3













