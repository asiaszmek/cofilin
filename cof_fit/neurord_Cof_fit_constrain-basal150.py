#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''trying to fit cof to bosh data '''


import ajustador as aju
import numpy as np
from ajustador import drawing,loadconc,nrd_fitness
from ajustador.helpers import converge,save_params
import os

dirname='cof_opt_150'
#name of model xml file for optimization
model_set='Model_Cof_fit'
#name of experimental data, a simulation file in this case
exp_name='Bosch_Hedrick_cof-basal150'
#molecule to compare between 'experiments' and simulations
mol={'RacGTP': ['RacGTP'],'Cofactin':['Cofactin']}#['RacGTP','Cofactin']
#directory to store output during optimization
tmpdir='/tmp/'+dirname
start_stim=90 # time stim start sec 
norm_method ='percent'


# number of iterations, use 1 for testing
iterations=100
# default popsize=8, use 3 for testing
popsize=8
test_size=25#
rootdir='./' ### doesn't work if using at
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
#Double check. #2
params = aju.optimize.ParamSet(P('Racact_fwd_rate',2.79888e-07, min=2.8e-9, max=2.8e-5, xpath='//Reaction[@id="RacGDP+Kal--pKalRacGDP"]/forwardRate'),
                               P('Racact_bckd_rate',0.0001658, fixed='Racact_kcat_rate',constant=4, xpath='//Reaction[@id="RacGDP+Kal--pKalRacGDP"]/reverseRate'),
                               P('Racact_kcat_rate',5.91e-05, min=5.91e-7, max=5.91e-3, xpath='//Reaction[@id="pKalRacGDP--pKal+RacGTP"]/forwardRate'),
                               P('PAKact_fwd_rate',9.6e-06, min=9.6e-8, max=9.6e-4, xpath='//Reaction[@id="RacGTP+PAK--RacGTPPAK"]/forwardRate'),
                               P('PAKact_bckd_rate',0.0015, min=0.0015e-2, max=0.0015e2, xpath='//Reaction[@id="RacGTP+PAK--RacGTPPAK"]/reverseRate'),
                               P('LIMKact_fwd_rate',0.00485182, min=0.0049e-2, max=0.0049e2, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/forwardRate'),
                               P('LIMKact_bckd_rate',0.353238, fixed='LIMKact_kcat_rate',constant=4, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/reverseRate'),
                               P('LIMKact_kcat_rate',0.0875, min=0.088e-2, max=0.088e2, xpath='//Reaction[@id="LIMKRacPAK--RacPAK+pLIMK"]/forwardRate'),
                               P('SSHact_fwd_rate',5.31825e-06, min=5.32e-8, max=5.32e-4, xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/forwardRate'),
                               P('SSHact_bckd_rate',0.000487042,fixed='SSHact_kcat_rate',constant=4 , xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/reverseRate'),
                               P('SSHact_kcat_rate',0.000122, min=0.000122e-2, max=0.000122e2, xpath='//Reaction[@id="CaNCamCa4pSSH--SSH+CaNCamCa4"]/forwardRate'),
                               P('Cofact_fwd_rate',1.08022e-05, min=1.08e-7, max=1.08e-3, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/forwardRate'),
                               P('Cofact_bckd_rate',1.27473, fixed='Cofact_kcat_rate',constant=4, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/reverseRate'),
                               P('Cofact_kcat_rate',0.35, min=0.35e-2, max=0.35e2, xpath='//Reaction[@id="SSHpCof--Cof+SSH"]/forwardRate'),
                               P('actinact_fwd_rate',5e-09, min=5e-11, max=5e-7, xpath='//Reaction[@id="actin+Cof--Cofactin"]/forwardRate'),
                               P('actinact_bckd_rate',0.001, min=0.001e-2, max=0.001e2, xpath='//Reaction[@id="actin+Cof--Cofactin"]/reverseRate'),
                               P('LIMKinact_fwd_rate',0.000175181, min=0.00016e-2, max=0.00016e2, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/forwardRate'),
                               P('LIMKinact_bckd_rate',0.0087791, fixed='LIMKinact_kcat_rate',constant=4, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/reverseRate'),
                               P('LIMKinact_kcat_rate',0.0283655, min=0.028e-2, max=0.028e2, xpath='//Reaction[@id="SSHpLIMK--pLIMK+SSH"]/forwardRate'),
                               P('SSHinact_fwd_rate',0.000117539, min=0.00012e-2, max=0.00012e2, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/forwardRate'),
                               P('SSHinact_bckd_rate',0.0914291, fixed='SSHinact_kcat_rate',constant=4, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/reverseRate'),
                               P('SSHinact_kcat_rate',0.0225, min=0.0225e-2, max=0.0225e2, xpath='//Reaction[@id="SSHRacPAK--pSSH+RacPAK"]/forwardRate'),
                               P('Cofinact_fwd_rate',0.00130698, min=0.0013e-2, max=0.0013e2, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/forwardRate'),
                               P('Cofinact_bckd_rate',0.066559, fixed='Cofinact_kcat_rate',constant=4, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/reverseRate'),
                               P('Cofinact_kcat_rate',0.017, min=0.017e-2, max=0.017e2, xpath='//Reaction[@id="pLIMKCof--pCof+pLIMK"]/forwardRate'))

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

'''
P('Racact_fwd_rate',2.79888e-07, min=2.8e-9, max=2.8e-5, xpath='//Reaction[@id="RacGDP+Kal--pKalRacGDP"]/forwardRate'),
                               P('Racact_bckd_rate',0.0001658, fixed='Racact_kcat_rate',constant=4, xpath='//Reaction[@id="RacGDP+Kal--pKalRacGDP"]/reverseRate'),
                               P('Racact_kcat_rate',5.91e-05, min=5.91e-7, max=5.91e-3, xpath='//Reaction[@id="pKalRacGDP--pKal+RacGTP"]/forwardRate'),
                               P('PAKact_fwd_rate',9.6e-06, min=9.6e-8, max=9.6e-4, xpath='//Reaction[@id="RacGTP+PAK--RacGTPPAK"]/forwardRate'),
                               P('PAKact_bckd_rate',0.0015, min=0.0015e-2, max=0.0015e2, xpath='//Reaction[@id="RacGTP+PAK--RacGTPPAK"]/reverseRate'),
                               P('LIMKact_fwd_rate',0.00485182, min=0.0049e-2, max=0.0049e2, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/forwardRate'),
                               P('LIMKact_bckd_rate',0.353238, fixed='LIMKact_kcat_rate',constant=4, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/reverseRate'),
                               P('LIMKact_kcat_rate',0.0875, min=0.088e-2, max=0.088e2, xpath='//Reaction[@id="LIMKRacPAK--RacPAK+pLIMK"]/forwardRate'),
                               P('SSHact_fwd_rate',5.31825e-06, min=5.32e-8, max=5.32e-4, xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/forwardRate'),
                               P('SSHact_bckd_rate',0.000487042,fixed='SSHact_kcat_rate',constant=4 , xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/reverseRate'),
                               P('SSHact_kcat_rate',0.000122, min=0.000122e-2, max=0.000122e2, xpath='//Reaction[@id="CaNCamCa4pSSH--SSH+CaNCamCa4"]/forwardRate'),
                               P('Cofact_fwd_rate',1.08022e-05, min=1.08e-7, max=1.08e-3, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/forwardRate'),
                               P('Cofact_bckd_rate',1.27473, fixed='Cofact_kcat_rate',constant=4, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/reverseRate'),
                               P('Cofact_kcat_rate',0.35, min=0.35e-2, max=0.35e2, xpath='//Reaction[@id="SSHpCof--Cof+SSH"]/forwardRate'),
                               P('actinact_fwd_rate',5e-09, min=5e-11, max=5e-7, xpath='//Reaction[@id="actin+Cof--Cofactin"]/forwardRate'),
                               P('actinact_bckd_rate',0.001, min=0.001e-2, max=0.001e2, xpath='//Reaction[@id="actin+Cof--Cofactin"]/reverseRate'),
                               P('LIMKinact_fwd_rate',0.000175181, min=0.00016e-2, max=0.00016e2, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/forwardRate'),
                               P('LIMKinact_bckd_rate',0.0087791, fixed='LIMKinact_kcat_rate',constant=4, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/reverseRate'),
                               P('LIMKinact_kcat_rate',0.0283655, min=0.028e-2, max=0.028e2, xpath='//Reaction[@id="SSHpLIMK--pLIMK+SSH"]/forwardRate'),
                               P('SSHinact_fwd_rate',0.000117539, min=0.00012e-2, max=0.00012e2, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/forwardRate'),
                               P('SSHinact_bckd_rate',0.0914291, fixed='SSHinact_kcat_rate',constant=4, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/reverseRate'),
                               P('SSHinact_kcat_rate',0.0225, min=0.0225e-2, max=0.0225e2, xpath='//Reaction[@id="SSHRacPAK--pSSH+RacPAK"]/forwardRate'),
                               P('Cofinact_fwd_rate',0.00130698, min=0.0013e-2, max=0.0013e2, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/forwardRate'),
                               P('Cofinact_bckd_rate',0.066559, fixed='Cofinact_kcat_rate',constant=4, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/reverseRate'),
                               P('Cofinact_kcat_rate',0.017, min=0.017e-2, max=0.017e2, xpath='//Reaction[@id="pLIMKCof--pCof+pLIMK"]/forwardRate'))
'''
