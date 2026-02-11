

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''trying to fit cof to bosh data '''


import ajustador as aju
import numpy as np
from ajustador import drawing,loadconc,nrd_fitness
from ajustador.helpers import converge,save_params
import os

dirname='cof_opt_full'
#name of model xml file for optimization
model_set='Model_Cof_fitfull.xml'
#name of experimental data, a simulation file in this case
exp_name='Bosch_Hedrick_cof-basal300'
#molecule to compare between 'experiments' and simulations
mol={'RacPAK':['RacPAK','RacGTP'],'Cofactin':['Cof', 'Cofactin']}#mol={'RacGTP': ['RacGTP'],'Cof':['Cof','Cofactin']}
#directory to store output during optimization
tmpdir='/tmp/'+dirname+'newversionNEW'
start_stim=30 # time stim start sec 
#norm_method ='percent'


# number of iterations, use 1 for testing
iterations=100
# default popsize=8, use 3 for testing
popsize=8
test_size=25
rootdir='./' 
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
#use #2 exp since loading exp 
#exp = aju.xml.NeurordResult(exp_set)
#this command indicates that experimental data are concentration in csv formatted files
os.chdir(dirname)
exp=loadconc.CSV_conc_set(exp_name,
                          stim_time=start_stim)
#print('***********************************',exp.data[0].waves['Cof'].wave)
#specify parameters to vary, either from ReactionScheme or InitialConditions
P = aju.xml.XMLParam
params = aju.optimize.ParamSet(P('Racact_fwd_rate',8.4e-8, min=9e-10, max=9e-6, xpath='//Reaction[@id="RacGDP+Kal--pKalRacGDP"]/forwardRate'),
                               P('Racact_bckd_rate',0.00011, fixed='Racact_kcat_rate',constant=4, xpath='//Reaction[@id="RacGDP+Kal--pKalRacGDP"]/reverseRate'),
                               P('Racact_kcat_rate',2.7e-5, min=3e-7, max=3e-3, xpath='//Reaction[@id="pKalRacGDP--pKal+RacGTP"]/forwardRate'),
                               P('Kalinac_fwd_rate',0.00005e-3, min=0.00005e-5, max=0.00005e-1, xpath='//Reaction[@id="pKal+PP1--pKalPP1"]/forwardRate'),
                               P('Kalinac_bckd_rate',0.04e-3, fixed='Kalinac_kcat_rate',constant=4, xpath='//Reaction[@id="pKal+PP1--pKalPP1"]/reverseRate'),
                               P('Kalinac_kcat_rate',0.01e-3, min=0.01e-5, max=0.01e-1, xpath='//Reaction[@id="pKalPP1--Kal+PP1"]/forwardRate'),
                               P('PAKact_fwd_rate',0.0094e-3, min=0.0094e-5, max=0.0094e-1, xpath='//Reaction[@id="RacGTP+PAK--RacGTPPAK"]/forwardRate'),
                               P('PAKact_bckd_rate',0.00082, min=0.00082e-2, max=0.00082e2, xpath='//Reaction[@id="RacGTP+PAK--RacGTPPAK"]/reverseRate'),
                               P('PAKin_fwd_rate',4.676e-7, min=4.676e-9, max=4.676e-5,xpath='//Reaction[@id="RacPAK--RacGDP+PAK"]/forwardRate'),
                               P('LIMKact_fwd_rate',0.103e-3, min=0.103e-5, max=0.103e-1, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/forwardRate'),
                               P('LIMKact_bckd_rate',0.333, fixed='LIMKact_kcat_rate',constant=4, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/reverseRate'),
                               P('LIMKact_kcat_rate',0.083, min=0.083e-2, max=0.083e2, xpath='//Reaction[@id="LIMKRacPAK--RacPAK+pLIMK"]/forwardRate'),
                               P('SSHact_fwd_rate',4.89e-5, min=4.89e-7, max=4.89e-3, xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/forwardRate'),
                               P('SSHact_bckd_rate',0.0005,fixed='SSHact_kcat_rate',constant=4 , xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/reverseRate'),
                               P('SSHact_kcat_rate',0.00012, min=0.00012e-2, max=0.00012e2, xpath='//Reaction[@id="CaNCamCa4pSSH--SSH+CaNCamCa4"]/forwardRate'),
                               P('Cofact_fwd_rate',2.8e-3, min=2.8e-5, max=2.8e-1, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/forwardRate'),
                               P('Cofact_bckd_rate',0.09, fixed='Cofact_kcat_rate',constant=4, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/reverseRate'),
                               P('Cofact_kcat_rate',0.22, min=0.22e-2, max=0.22e2, xpath='//Reaction[@id="SSHpCof--Cof+SSH"]/forwardRate'),
                               P('actinact_fwd_rate',1.2e-7, min=1.2e-9, max=1.2e-5, xpath='//Reaction[@id="actin+Cof--Cofactin"]/forwardRate'),
                               P('actinact_bckd_rate',0.009, min=0.009e-2, max=0.009e2, xpath='//Reaction[@id="actin+Cof--Cofactin"]/reverseRate'),
                               P('LIMKinact_fwd_rate',0.0017, min=0.0017e-2, max=0.0014e2, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/forwardRate'),
                               P('LIMKinact_bckd_rate',2.78, fixed='LIMKinact_kcat_rate',constant=4, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/reverseRate'),
                               P('LIMKinact_kcat_rate',0.69, min=0.69e-2, max=0.69e2, xpath='//Reaction[@id="SSHpLIMK--pLIMK+SSH"]/forwardRate'),
                               P('SSHinact_fwd_rate',0.103e-3, min=0.103e-5, max=0.103e-1, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/forwardRate'),
                               P('SSHinact_bckd_rate',0.33, fixed='SSHinact_kcat_rate',constant=4, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/reverseRate'),
                               P('SSHinact_kcat_rate',0.083, min=0.083e-2, max=0.083e2, xpath='//Reaction[@id="SSHRacPAK--pSSH+RacPAK"]/forwardRate'),
                               P('Cofinact_fwd_rate',0.011, min=0.011e-2, max=0.011e2, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/forwardRate'),
                               P('Cofinact_bckd_rate',0.87, fixed='Cofinact_kcat_rate',constant=4, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/reverseRate'),
                               P('Cofinact_kcat_rate',0.22, min=0.22e-2, max=0.22e2, xpath='//Reaction[@id="pLIMKCof--pCof+pLIMK"]/forwardRate'))

#this command indicates that experiments are from a previous simulation
###################### END CUSTOMIZATION #######################################

fitness = nrd_fitness.specie_concentration_fitness(species_list=mol,start=start_stim)

fit = aju.optimize.Fit(tmpdir, exp, model_set, None, fitness, params,
                       _make_simulation=aju.xml.NeurordSimulation.make,
                       _result_constructor=aju.xml.NeurordResult)
fit.load()
fit.do_fit(iterations, sigma=0.3,popsize=popsize)
save_params.save_params(fit,0,1)
mean_dict,std_dict,CV=converge.iterate_fit(fit,test_size,popsize)
########################################### Done with fitting

#to look at centroid [0] or stdev [6] of cloud of good results:
if callable(fit.optimizer.result):
    result=fit.optimizer.result()
else:
    result=fit.optimizer.result
for i,p in enumerate(fit.params.unscale(result[0])):
        print(fit.param_names()[i],'=',p, '+/-', fit.params.unscale(result[6])[i])


#to save
save_params.save_params(fit,0,1)

#to look at fit history
#aju.drawing.plot_history(fit,fit.measurement,mol_dict=mol)





