
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

plt.rcdefaults()
fig,ax = plt.subplots()

from SALib.sample import saltelli
from SALib.analyze import sobol

param_ranges = {
    'num_vars': 8,
    'names': ['R',
              'MU',
              'G',
              'SIG',
              'INBR',
              'BETA',
              'XI',
              'K'],
    'bounds': [[20, 80],
               [0.01, 0.4],
               [0.5, 0.99],
               [0.01, 0.3],
               [0.5, 0.99],
               [0.01, 2],
               [0.5, 0.99],
               [50, 150]]
}

# Generate samples
param_values = saltelli.sample(param_ranges, 1024)

Y = np.zeros([param_values.shape[0]])

def evaluate_model(pars):
    # Defining alleles and genotypes
    # R = pars[0]
    # MU = pars[1]
    # G = pars[2]
    # SIG = pars[3]
    # INBR = pars[4]
    # BETA = pars[5]
    # XI = pars[6]
    # K = pars[7]

    alleles = ["A", "B", "G"]
    N_ALLELES = len(alleles)
    allele_growth_rates = [pars[0], pars[0]*0.9, pars[0]*0.8] # B's growth rate is 90% of A. G's is 80% of A
    allele_resistances = [0, pars[6], pars[6]] # B and G assumed to convey the same resistance
    genotypes = []
    genotype_growth_rates = []
    genotype_resistances = []
    for i in range(0, N_ALLELES):
        for j in range(i, N_ALLELES):
            genotypes.append(alleles[i]+alleles[j])
            genotype_growth_rates.append(np.mean([allele_growth_rates[i],
                                                  allele_growth_rates[j]]))
            genotype_resistances.append(np.mean([allele_resistances[i],
                                                 allele_resistances[j]]))
    N_GENOTYPES = len(genotypes)

    # Population dynamics setup
    # Number of snail generations to track. Default = 100
    N_GENS = 100

    # Define initial (starting) snail population for the genotypes [AA, AB, AG, BB, BG, GG]
    # Assuming we seed a population with 1 gene drive individual, Default = [49, 20, 0, 30, 0, 1]
    initial_pop = [49, 20, 0, 30, 0, 1]

    # Define the population genetics matrix. We'll use this to graph out the population over several generations
    population_genetics = np.zeros([N_GENOTYPES, N_GENS])

    # Initialize the population
    population_genetics[:,0] = initial_pop

    # Creating an empty array for the number of infected snails
    infected_snails = np.zeros((N_GENS, N_GENOTYPES))
    for x in range(0, N_GENOTYPES):
        infected_snails[0][x] = (1 - pars[1]) * pars[5] * (1 - genotype_resistances[x]) * initial_pop[x]

    # View characteristics of the genotypes we are working with
    gdf = pd.DataFrame({"Growth Rate": genotype_growth_rates,
                       "Resistance": genotype_resistances})
    gdf.index = genotypes

    def get_transition_matrix(genotypes, previous_gen_proportions):
        # Framing the proportions in a dataframe we can call by genotype
        proportions = pd.DataFrame(previous_gen_proportions, index = genotypes)

        # Approximating small values as 0 to help with runtime
        for genotype in genotypes:
            if proportions.loc[genotype][0] < 1e-6:
                proportions.loc[genotype][0] = 0
        
        # Outcrossing transition matrix setup
        out_tm = np.zeros((N_GENOTYPES, N_GENOTYPES))
        out_tm = pd.DataFrame(out_tm, index = genotypes, columns = genotypes)
        
        # Generating the outcrossing transition matrix
        # genotype1 x genotype 2 --> genotype_nextgen
        # Rows: genotype1
        # Columns: genotype_nextgen
        for genotype1 in genotypes:
            for genotype2 in genotypes:
                for allele1 in genotype1:
                    for allele2 in genotype2:
                        genotype_nextgen = allele1 + allele2
                        genotype_nextgen = "".join(sorted(genotype_nextgen)) # alphabetize the genotype
                        rate = np.mean([gdf.loc[genotype1]["Growth Rate"], gdf.loc[genotype2]["Growth Rate"]])

                        # Approximating small values as 0 to help with runtime
                        if rate < 1e-6:
                            rate=0
                            
                        # Gene Drive Condition:
                        # If the nextgen genotype is heterozygous for the gene drive,
                        # then the efficiency rate g determines the outcomes
                        # - Most form a gene drive genotype 'GG' at a rate g
                        # - Some stay in the nextgen genotype at a rate (1-g)
                        if bool('G' in allele1) != bool('G' in allele2):
                            out_tm["GG"][genotype1] = out_tm["GG"][genotype1] + pars[2] * (1 / 4) * rate * proportions.loc[genotype1][0] * proportions.loc[genotype2][0]
                            out_tm[genotype_nextgen][genotype1] = out_tm[genotype_nextgen][genotype1] + (1 - pars[2]) * (1 / 4) * rate * proportions.loc[genotype1][0] * proportions.loc[genotype2][0]
                        else:
                            out_tm[genotype_nextgen][genotype1] = out_tm[genotype_nextgen][genotype1] + (1 / 4) * rate * proportions.loc[genotype1][0] * proportions.loc[genotype2][0]
        
        # Selfing (asexual reproduction) transition matrix setup
        self_tm = np.zeros((N_GENOTYPES, N_GENOTYPES))
        self_tm = pd.DataFrame(self_tm, index = genotypes, columns = genotypes)

        # Generating the selfing transition matrix
        # genotype_self x genotype_self --> genotype_nextgen
        # Rows: genotype_self
        # Columns: genotype_nextgen
        for genotype_self in genotypes:
            for allele1 in genotype_self:
                for allele2 in genotype_self:
                
                    # Find the genotype we're working with and alphabetize it
                    genotype_nextgen = allele1 + allele2
                    genotype_nextgen = "".join(sorted(genotype_nextgen))
                    rate = gdf.loc[genotype_self]["Growth Rate"]

                    # Approximating small values as 0 to help with runtime
                    if rate < 1e-6:
                        rate=0
                
                    # Gene Drive Condition applies
                    if bool ('G' in allele1) != bool('G' in allele2):
                        self_tm['GG'][genotype_self] = self_tm['GG'][genotype_self] + pars[2] * (1 / 4) * rate * proportions.loc[genotype_self][0]
                        self_tm[genotype_nextgen][genotype_self] = self_tm[genotype_nextgen][genotype_self] + (1 - pars[2]) * (1 / 4) * rate * proportions.loc[genotype_self][0]
                    else:
                        self_tm[genotype_nextgen][genotype_self] = self_tm[genotype_nextgen][genotype_self] + (1 / 4) * rate * proportions.loc[genotype_self][0]
        
        # Combine the two transition matrices, applying the selfing rate and the inbreeding cost
        tm = (1 - pars[3]) * out_tm + pars[3] * pars[4] * self_tm
        
        return tm

    # Chunk 4
    # Starting with index of 1 because we have generation 0 already defined
    for i in range(1, N_GENS):    
        # Apply natural death and infection-caused castration to get to the reproducing pool of snails
        reproducing_population = np.zeros(N_GENOTYPES)
        for x in range(0, N_GENOTYPES):
                infected_snails[i][x] = (1 - pars[1]) * pars[5] * (1 - gdf.loc[genotypes[x]]["Resistance"]) * population_genetics[x, i-1]
                if infected_snails[i][x] < 1e-6:
                    infected_snails[i][x] = 0
                reproducing_population[x] = population_genetics[x, i-1] - pars[1] * population_genetics[x, i-1] - infected_snails[i][x]

        # The last element is the total number of snails in the population
        total_repr_pop = np.sum(reproducing_population)

        if total_repr_pop < 1e-6:
            genotype_proportions = np.zeros(N_GENOTYPES)
        else:
            genotype_proportions = reproducing_population/total_repr_pop
        
        tm =  get_transition_matrix(genotypes, genotype_proportions)

        # Find total amount of new growth, by genotype, and in total
        genotype_growth = np.sum(tm)
        growth_total = np.sum(genotype_growth)

        # apply a coefficient for logistic growth
        if total_repr_pop < 1e-6:
            log_growth_coeff = pars[7]
        else:
            log_growth_coeff = total_repr_pop * pars[7] / (total_repr_pop + (pars[7] - total_repr_pop) * np.exp(-np.clip(growth_total, 0, 1e6))) - total_repr_pop
        
        # new population = reproducing (surviving) population + new births
        if (growth_total*log_growth_coeff) < 1e-6:
            new_pop = np.zeros(N_GENOTYPES)
        else:
            new_pop = reproducing_population + genotype_growth/growth_total*log_growth_coeff
        
        population_genetics[:,i] = new_pop


    return np.sum(infected_snails[99,:])


for i, X in enumerate(param_values):
    Y[i] = evaluate_model(X)

Si = sobol.analyze(param_ranges, Y)

y_pos = np.arange(len(param_ranges['names']))

# plots a bar graph 
hbars = ax.barh(y_pos, Si['ST'], align = 'center')
ax.set_yticks(y_pos)
ax.set_yticklabels(param_ranges['names'])
ax.invert_yaxis()
ax.bar_label(hbars, fmt='%.2f')
ax.set_ylabel('Parameters')
ax.set_xlabel('Total Sensitivity Index')
plt.show()

