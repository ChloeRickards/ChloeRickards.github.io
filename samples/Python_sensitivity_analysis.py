# Python sample - Sensitivity Analysis

# This code implements a sensitivity analysis to find the most sensitive parameters for 
# a model of a population of freshwater snails, the intermediate host for a human
# infectious disease called schistosomiasis. This analysis goes over thousands of
# combinations of parameter values to find the ones that most highly impact
# the desired output. Here, the desired output is the number of snails infected
# with schistosomiasis. Since this model is investigating the impacts
# of a gene drive, the most sensitive parameters could determine how forceful
# the gene drive needs to be or which environments to apply the gene drive.

# I chose a Sobol sensitivity analysis here, which is a variance-based sensitivity
# analysis. It takes both first- and second- order parameter interactions into account.
# Here, I'm interested in the total-effect index for each parameter, which weighs
# both orders analyzed. The resulting total-effect indices are plotted.

# You can see the full implementation of this model (including the results of
# the sensitivity analysis for this model) here:
# https://chloerickards.github.io/genedrive/


# Import sensitivity analysis packages
from SALib.sample import saltelli
from SALib.analyze import sobol

# This defines the parameters we are testing and the range of values we will be testing them in
param_ranges = {
    'num_vars': 8,
    'names': ['R',
              'MU',
              'GAMMA',
              'SIG',
              'INBR',
              'BETA',
              'XI',
              'K'],
    'bounds': [[20, 80], # per capita fertility rate [default = 54]
               [0.01, 0.4], # natural death rate [default = 0.2]
               [0.5, 0.99], # gene drive efficiency [default = 0.95] 0.5 is regular Mendelian inheritance rate
               [0.01, 0.3], # selfing rate [default = 0.15]
               [0.5, 0.99], # inbreeding fitness cost [default = 0.85]
               [0.01, 2], # per capita infection rate [default = 0.5]
               [0.5, 0.99], # level of resistance conferred by the B and G alleles [default = 0.8]
               [50, 150]] # carrying capacity [default = 100]
}

# Generate parameter samples. Note - 1024 samples makes the evaluate_model section run for a VERY long time on a home computer
param_values = saltelli.sample(param_ranges, 1024)

# This creates a placeholder array for the output
# Here, our output is the number of infected snails at the end of the simulation.
Y = np.zeros([param_values.shape[0]])

def evaluate_model(pars):
    # This is where the model would go. It's excluded here for brevity, but the full model can be seen on my Github
    # https://github.com/ChloeRickards/chloerickards.github.io/blob/master/files/sensitivity-analysis-snails.py
    # 
    # :param: pars - array of parameter values, sampled from the ranges given above.
    # :return: float - np.sum(infected_snails[99,:]) - The number of infected snails by the end of the model
    # 
    return np.sum(infected_snails[99,:])

# Evaluate model for each array of parameter values
for i, X in enumerate(param_values):
    Y[i] = evaluate_model(X)

# Perform the sensitivity analysis
Si = sobol.analyze(param_ranges, Y)

# plots a bar graph of the total sensitivity index
y_pos = np.arange(len(param_ranges['names']))
hbars = ax.barh(y_pos, Si['ST'], align = 'center')
ax.set_yticks(y_pos)
ax.set_yticklabels(param_ranges['names'])
ax.invert_yaxis()
ax.bar_label(hbars, fmt='%.2f')
ax.set_ylabel('Parameters')
ax.set_xlabel('Total Sensitivity Index')
plt.show()