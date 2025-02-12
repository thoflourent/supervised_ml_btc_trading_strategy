import pandas as pd
import numpy as np
from constants import ALPHA_PTH, BETA_PTH, PCT_PERIOD


class LabelCreation:

    @staticmethod
    def get_alpha_and_beta(data, alpha_pth=ALPHA_PTH, beta_pth=BETA_PTH, pct_period=PCT_PERIOD):
        pct_change = data['Close'].pct_change()

        alpha = pct_change.abs().quantile(alpha_pth)
        beta = pct_change.abs().quantile(beta_pth)

        return alpha, beta


    @staticmethod
    def assign_labels(data, b_window, f_window, alpha, beta):
        x = data.copy()
        x['Close_MA'] = x['Close'].ewm(span=b_window).mean()
        x['s-1'] = x['Close'].shift(-1 * f_window)
        x['alpha'] = alpha
        x['beta'] = beta * (1 + (f_window * 0.1))
        x['label'] = x.apply(LabelCreation.check_label, axis=1)
        return x['label']


    @staticmethod
    def check_label(z):
        if (abs((z['s-1'] - z['Close_MA']) / z['Close_MA']) > z['alpha']) and \
                (abs((z['s-1'] - z['Close_MA']) / z['Close_MA']) < (z['beta'])):
            if z['s-1'] > z['Close_MA']:
                return -1
            elif z['s-1'] < z['Close_MA']:
                return 1
            else:
                return 0
        else:
            return 0

