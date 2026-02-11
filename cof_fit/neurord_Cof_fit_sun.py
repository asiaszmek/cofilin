#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''trying to fit cof to bosh data '''


import ajustador as aju
import numpy as np
from ajustador import drawing,loadconc,nrd_fitness
from ajustador.helpers import converge,save_params
import os

dirname='cof_opt'
#name of model xml file for optimization
model_set='Model_Cof_fit'
#name of experimental data, a simulation file in this case
exp_name='Bosch_cof'
#molecule to compare between 'experiments' and simulations
mol=['Cof']
#directory to store output during optimization
tmpdir='/tmp/'+dirname#how again
start_stim=36 # time stim start sec 
#norm_method ='percent'


# number of iterations, use 1 for testing
iterations=1#100
# default popsize=8, use 3 for testing
popsize=3#8
test_size=0#
rootdir='./' ### doesn't work if using at
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
#use #2 exp since loading exp 
#exp = aju.xml.NeurordResult(exp_set)
#this command indicates that experimental data are concentration in csv formatted files
os.chdir(dirname)
exp=loadconc.CSV_conc_set(exp_name,
                          stim_time=start_stim)
print('***********************************',exp.data[0].waves['Cof'].wave)
#specify parameters to vary, either from ReactionScheme or InitialConditions
P = aju.xml.XMLParam
#Double check. #2
params = aju.optimize.ParamSet(P('limkact_fwd_rate',4.92499e-5, min=5e-8, max=5e-3, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/forwardRate'),
                               P('limkact_bak_rate',1.11691, min=1.5e-3, max=1.5e3, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/reverseRate'),
                               P('limkact_kcat_rate',0.0397626, min=0.04e-3, max=0.04e3, xpath='//Reaction[@id="LIMKRacPAK--RacPAK+pLIMK"]/forwardRate'),
                               P('limkinact_fwd_rate',0.000239654, min=0.0025e-3, max=0.0025e3, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/forwardRate'),
                               P('limkinact_bak_rate',0.000358413, min=0.0004e-3, max=0.0004e3, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/reverseRate'),
                               P('limkinact_kcat_rate',0.000236355, min=0.0002e-3, max=0.0002e3, xpath='//Reaction[@id="SSHpLIMK--pLIMK+SSH"]/forwardRate'),
                               P('SSHact_fwd_rate',0.000374011, min=0.0004e-3, max=0.0004e3, xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/forwardRate'),
                               P('SSHact_bak_rate',0.00993811, min=0.01e-3, max=0.01e3, xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/reverseRate'),
                               P('SSHact_kcat_rate',1.24298e-5, min=1.3e-8, max=1.3e-2, xpath='//Reaction[@id="CaNCamCa4pSSH--SSH+CaNCamCa4"]/forwardRate'),
                               P('SSHinact_fwd_rate',0.000205922, min=0.0002e-3, max=0.0002e3, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/forwardRate'),
                               P('SSHinact_bak_rate',0.00111703, min=0.0015e-3, max=0.0015e3, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/reverseRate'),
                               P('SSHinact_kcat_rate',7.02881e-5, min=7e-8, max=7e-2, xpath='//Reaction[@id="SSHRacPAK--pSSH+RacPAK"]/forwardRate'),
                               P('Cofact_fwd_rate',1.02418e-6, min=1e-9, max=1e-3, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/forwardRate'),
                               P('Cofact_bak_rate',0.188625, min=0.2e-3, max=0.2e3, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/reverseRate'),
                               P('Cofact_kcat_rate',0.00406818, min=0.004e-3, max=0.004e3, xpath='//Reaction[@id="SSHpCof--Cof+SSH"]/forwardRate'),
                               P('Cofinact_fwd_rate',0.000336696, min=0.0004e-3, max=0.0004e3, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/forwardRate'),
                               P('Cofinact_bak_rate',0.175602, min=0.2e-3, max=0.2e3, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/reverseRate'),
                               P('Cofinact_kcat_rate',1e-7, min=1e-10, max=1e-4, xpath='//Reaction[@id="pLIMKCof--pCof+pLIMK"]/forwardRate'))

#this command indicates that experiments are from a previous simulation
###################### END CUSTOMIZATION #######################################

fitness = nrd_fitness.specie_concentration_fitness(species_list=mol)

fit = aju.optimize.Fit(tmpdir, exp, model_set, None, fitness, params,
                       _make_simulation=aju.xml.NeurordSimulation.make,
                       _result_constructor=aju.xml.NeurordResult)
fit.load()
fit.do_fit(iterations, sigma=0.3)
mean_dict,std_dict,CV=converge.iterate_fit(fit,test_size,popsize)
########################################### Done with fitting

#to look at centroid [0] or stdev [6] of cloud of good results:
if callable(fit.optimizer.result):
    result=fit.optimizer.result()
else:
    result=fit.optimizer.result
for i,p in enumerate(fit.params.unscale(result[0])):
        print(fit.param_names()[i],'=',p, '+/-', fit.params.unscale(result[6])[i])

#to look at fit history
aju.drawing.plot_history(fit,fit.measurement)

#to save
save_params.save_params(fit,0,1)

