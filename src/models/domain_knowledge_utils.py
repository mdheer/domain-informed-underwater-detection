import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import random
#%%
class DomainKnowledgeClassifier():
    '''
       This class performs motion trajectory-based classification. It compares a
       target object trajectory with the predicted trajectories from an existing model (mathematical/simulation)
       of certain classes and based on the comparison error it extracts the class probabilities.
       TODOs: The current implementation supports only one class. I need to extend it to more classes
       param:
           csv_trajectories_path: str, Path to csv with measured and predicted trajectories
           categories: list, the categories of the classes
       '''
    def __init__(self, json_trajectories_path, categories):
        self.main_dataframe = pd.read_json(json_trajectories_path)
        self.categories = categories

    def error_dk_measured_trajectory(self):
        """
        This method computes the error between measured and predicted trajectory of
        target objects
        TODOs: now the implementation only
        calculates the position error for one object. Add errors for many objects (pos and vel)
        :return:
            errors: dict, includes the MSE error between measured and predicted trajectory
        """
        errors = {}
        # for i in range(len(self.categories)):
        #     posbody_dm_key = f"posbody{i}_dm"
        #     posbody_dm_value = self.main_dataframe[f"posbody{i}_dm"]
        #     velbody_dm_key = f"velbody{i}_dm"
        #     velbody_dm_value  = self.main_dataframe[f"velbody{i}_dm"]
        #     posbody_measured_key = f"posbody{i}_measured"
        #     posbody_measured_value   = self.main_dataframe[f"posbody{i}_measured"]
        #     velbody_measured_key = f"velbody{i}_measured"
        #     velbody_measured_value= self.main_dataframe[f"velbody{i}_measured"]
        #     err_posbody_key = f"err_posbody_{i}"
        #     err_posbody_value = posbody_dm_value - posbody_measured_value
        #     err_velbody_key = f"err_velbody_{i}"
        #     err_velbody_value = velbody_dm_value - velbody_measured_value
        #
        #     errors[posbody_dm_key] = posbody_dm_value
        posbody_dm = self.main_dataframe['posbody_dm']
        posbody_measured = self.main_dataframe['posbody_measured']
        errors['mse_posbody'] = mean_squared_error(posbody_dm, posbody_measured)
        errors['rmse_posbody'] = np.sqrt(errors['mse_posbody'])
        return errors


    def probability_domain_knowledge_time_domain(self, errors, parameters):
        """
        This method returns the probability that a trajectory belongs to a specific target object
        based on the error between the measured trajectory and the predicted one.
        param:
            errors: dict with the calculated errors between the measured
            and the calculated trajectories
            parameters: dict: parameters of the probability of the error (mean, sigma)
        :return:
            p_posbody_dk: float the probability p(c|e) that a trajectory belongs to a specific object class
        """
        total_error = errors.sum()
        p_posbody_dk = (1/(parameters["sigma"]*math.sqrt(2*math.pi))) * np.exp(-(1/2)*((errors['mse_posbody']-parameters["mean"])**2/parameters["sigma"]**2))

        return p_posbody_dk

    # def probability_domain_knowledge_frequency_domain(self, errors, parameters):