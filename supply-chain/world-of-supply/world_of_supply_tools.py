import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

class SimulationTracker:
    def __init__(self, eposod_len, n_episods, facility_names):
        self.episod_len = eposod_len
        self.global_balances = np.zeros((n_episods, eposod_len))
        self.facility_names = list(facility_names)
        self.step_balances = np.zeros((n_episods, eposod_len, len(facility_names)))
    
    def add_sample(self, episod, t, global_balance, rewards):
        self.global_balances[episod, t] = global_balance
        assert self.facility_names == list(rewards.keys())                            # ensure facility order is preserved
        self.step_balances[episod, t, :] = np.array(list(rewards.values()))
        
    def render(self):
        fig, axs = plt.subplots(3, 1, figsize=(16, 12))
        x = np.linspace(0, self.episod_len, self.episod_len)
        axs[0].plot(x, self.global_balances.T)                                        # global balance
        axs[1].plot(x, np.cumsum(np.sum(self.step_balances, axis = 2), axis = 1).T )  # cumulative sum of rewards
        axs[2].plot(x, np.cumsum(self.step_balances[0, :, :], axis = 0))              # reward breakdown by facility for the first episod 
        axs[2].legend(self.facility_names, loc='upper left')
        plt.show()