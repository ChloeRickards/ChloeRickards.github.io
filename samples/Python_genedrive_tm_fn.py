# Python sample - gene drive transition matrix function

import numpy as np
import pandas as pd
import math

# This is a function for a population genetics model that takes in the genotype
# proportions from one generation and finds the rate at which the next generation
# of genotypes will be produced. This function was designed for a model of 
# a population of freshwater snails, the intermediate host for a human infectious
# disease called schistosomiasis. A simple application of a transition matrix for
# population genetics can be found here:
# https://juniperpublishers.com/bboaj/pdf/BBOAJ.MS.ID.555602.pdf

# This specific transition matrix adds two complexities. First, it takes place
# in an organism that reproduces both sexually and asexually. (This is governed
# by a "selfing" ratio - to study a system that only reproduces sexually, simply
# set this ratio to 0). Second, it applies gene drive mechanics where applicable.
# A gene drive favors the production of a certain genotype at very high rates. 

# You can see the full implementation of this model here:
# https://chloerickards.github.io/genedrive/


def get_transition_matrix(genotypes, gdf, previous_gen_proportions):
    # Obtain a transition matrix of genotype proportions from one generation to the next.
    # 
    # :param: genotypes - NumPy array - Names of the genotypes we are working with. Also the
    #                                   index for the gdf, below
    # :param: gdf - pandas DataFrame - Genotype DataFrame containing the names (as the index),
    #                                  growth rates, and other characteristics as needed
    #                                  for the genotypes we are working with. A gene drive
    #                                  genotype is denoted by a 'G' in the genotype name.
    # :param: previous_gen_proportions - NumPy array - Proportions corresponding to each
    #                                                  genotype, from the previous generation.
    # :return: tm - transition matrix containing the likelihood of the previous generation's
    #               genotypes (given by rows) generating offspring of another genotype
    #               (given by columns).
    """

    # Framing the previous generation's genotype proportions in a dataframe we can call by genotype name
    proportions = pd.DataFrame(previous_gen_proportions, index = genotypes)
    
    # Outcrossing (sexual reproduction) transition matrix setup
    out_tm = np.zeros((N_GENOTYPES, N_GENOTYPES))
    out_tm = pd.DataFrame(out_tm, index = genotypes, columns = genotypes)
    
    # Generating the outcrossing transition matrix by iterating through the
    # genotype combinations with a set of nested 'for' loops. 
    #
    # The logic here follows a Punnett square
    # genotype1 x genotype 2 --> genotype_nextgen
    # Rows: genotype1 (previous gen)
    # Columns: genotype_nextgen
    for genotype1 in genotypes:
        for genotype2 in genotypes:
            for allele1 in genotype1:
                for allele2 in genotype2:

                    # Find the genotype we're working with and alphabetize it
                    genotype_nextgen = allele1 + allele2
                    genotype_nextgen = "".join(sorted(genotype_nextgen))
                    rate = np.mean([gdf.loc[genotype1]["Growth Rate"], gdf.loc[genotype2]["Growth Rate"]])
                        
                    # Gene Drive Condition:
                    # If the nextgen genotype is heterozygous for the gene drive,
                    # then the efficiency rate g determines the outcomes
                    # - Most form a gene drive genotype 'GG' at a rate GAMMA
                    # - Some stay in the nextgen genotype at a rate (1 - GAMMA)
                    if bool('G' in allele1) != bool('G' in allele2):
                        out_tm["GG"][genotype1] = out_tm["GG"][genotype1] + GAMMA * (1 / 4) * rate * proportions.loc[genotype1][0] * proportions.loc[genotype2][0]
                        out_tm[genotype_nextgen][genotype1] = out_tm[genotype_nextgen][genotype1] + (1 - GAMMA) * (1 / 4) * rate * proportions.loc[genotype1][0] * proportions.loc[genotype2][0]
                    else:
                        out_tm[genotype_nextgen][genotype1] = out_tm[genotype_nextgen][genotype1] + (1 / 4) * rate * proportions.loc[genotype1][0] * proportions.loc[genotype2][0]
    
    # Selfing (asexual reproduction) transition matrix setup
    self_tm = np.zeros((N_GENOTYPES, N_GENOTYPES))
    self_tm = pd.DataFrame(self_tm, index = genotypes, columns = genotypes)

    # Generating the selfing transition matrix with another set of nested 'for' loops
    # 
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
            
                # Gene Drive Condition applies
                if bool ('G' in allele1) != bool('G' in allele2):
                    self_tm['GG'][genotype_self] = self_tm['GG'][genotype_self] + GAMMA * (1 / 4) * rate * proportions.loc[genotype_self][0]
                    self_tm[genotype_nextgen][genotype_self] = self_tm[genotype_nextgen][genotype_self] + (1 - GAMMA) * (1 / 4) * rate * proportions.loc[genotype_self][0]
                else:
                    self_tm[genotype_nextgen][genotype_self] = self_tm[genotype_nextgen][genotype_self] + (1 / 4) * rate * proportions.loc[genotype_self][0]
    
    # Combine the two transition matrices, applying the selfing rate and the inbreeding cost
    # Selfing rate = SIG (0 = no selfing, 1 = only selfing)
    # Cost of inbreeding = INBR (reduces the number of individuals produced by selfing by a fraction)
    tm = (1 - SIG) * out_tm + SIG * INBR * self_tm
    
    return tm

# Sample implementation of the transition matrix function, including setup and application

# 1. Setup and starting parameters

# Intrinsic growth rate (clutch size), per snail per generation
# Source = Costa 2004
R = 54

# Gene drive efficiency
# Source = Hammond et al. 2016
GAMMA = 0.95

# Defining alleles
alleles = ["A", "B", "G"] 
N_ALLELES = len(alleles)
allele_growth_rates = [R, R*0.9, R*0.8] # B's growth rate is 90% of A. G's is 80% of A

# Defining genotypes from alleles
genotypes = []
genotype_growth_rates = []
for i in range(0, N_ALLELES):
    for j in range(i, N_ALLELES):
        genotypes.append(alleles[i]+alleles[j])
        genotype_growth_rates.append(np.mean([allele_growth_rates[i],
                                              allele_growth_rates[j]]))

# Genotype Data Frame to hold the relevant genotype characteristics, indexed by genotype
gdf = pd.DataFrame({"Growth Rate": genotype_growth_rates}) # more characteristics besides genotype can be added as needed
gdf.index = genotypes

# Define initial (starting) snail population proportions for the genotypes [AA, AB, AG, BB, BG, GG]
# Assuming we seed a population with 1 gene drive individual
initial_pop = [49, 20, 0, 30, 0, 1]
initial_pop_proportions = initial_pop/np.sum(initial_pop)

# 2. Implement the function

tm =  get_transition_matrix(genotypes, gdf, initial_pop_proportions)

# 3. Apply results

# Find total amount of new growth, by genotype, and in total
genotype_growth = np.sum(tm)
growth_total = np.sum(genotype_growth)

# new population = reproducing (surviving) population + new births
# found via vector addition
new_pop = initial_pop + genotype_growth/growth_total

# repeat for several iterations to see genotype changes over multiple generations