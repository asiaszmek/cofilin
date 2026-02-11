#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''trying to fit cof to bosh data '''


import ajustador as aju
import numpy as np
from ajustador import drawing,loadconc,nrd_fitness
from ajustador.helpers import converge,save_params
import os

dirname='cof_opt_basal300'
#name of model xml file for optimization
model_set='Model_Cof_fitbasal.xml'
#name of experimental data, a simulation file in this case
exp_name='Nadia_cof-basal300'
#molecule to compare between 'experiments' and simulations
mol={'RacGTP':['RacGTP'],'Cofactin':['Cof', 'Cofactin'],'pCof':['pCof']}#mol={'RacGTP': ['RacGTP'],'Cof':['Cof','Cofactin']}
#directory to store output during optimization
tmpdir='/tmp/'+dirname+'basal'
start_stim=0 # time stim start sec 
#norm_method ='percent'


# number of iterations, use 1 for testing
iterations=100
# default popsize=8, use 3 for testing
popsize=8
test_size=25#25#
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
params = aju.optimize.ParamSet(P('Racact_fwd_rate',1.94e-6, min=2e-8, max=2.8e-4, xpath='//Reaction[@id="RacGDP+Kal--pKalRacGDP"]/forwardRate'),
                               P('Racact_bckd_rate',1.99e-5, fixed='Racact_kcat_rate',constant=4, xpath='//Reaction[@id="RacGDP+Kal--pKalRacGDP"]/reverseRate'),
                               P('Racact_kcat_rate',2.98e-5, min=3e-7, max=3e-3, xpath='//Reaction[@id="pKalRacGDP--pKal+RacGTP"]/forwardRate'),
                               P('PAKact_fwd_rate',7.2e-6, min=7.e-8, max=7.2e-4, xpath='//Reaction[@id="RacGTP+PAK--RacGTPPAK"]/forwardRate'),
                               P('PAKact_bckd_rate',0.014, min=0.014e-2, max=0.014e2, xpath='//Reaction[@id="RacGTP+PAK--RacGTPPAK"]/reverseRate'),
                               P('LIMKact_fwd_rate',0.0022, min=0.0022e-2, max=0.0022e2, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/forwardRate'),
                               P('LIMKact_bckd_rate',0.21, fixed='LIMKact_kcat_rate',constant=4, xpath='//Reaction[@id="RacPAK+LIMK--LIMKRacPAK"]/reverseRate'),
                               P('LIMKact_kcat_rate',0.3, min=0.3e-2, max=0.3e2, xpath='//Reaction[@id="LIMKRacPAK--RacPAK+pLIMK"]/forwardRate'),
                               P('SSHact_fwd_rate',4.44e-5, min=4.5e-7, max=4.5e-3, xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/forwardRate'),
                               P('SSHact_bckd_rate',0.0018,fixed='SSHact_kcat_rate',constant=4 , xpath='//Reaction[@id="CaNCamCa4+pSSH--CaNCamCa4pSSH"]/reverseRate'),
                               P('SSHact_kcat_rate',0.0018, min=0.002e-2, max=0.002e2, xpath='//Reaction[@id="CaNCamCa4pSSH--SSH+CaNCamCa4"]/forwardRate'),
                               P('Cofact_fwd_rate',4.22e-5, min=4.2e-7, max=4.2e-3, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/forwardRate'),
                               P('Cofact_bckd_rate',0.99, fixed='Cofact_kcat_rate',constant=4, xpath='//Reaction[@id="pCof+SSH--SSHpCof"]/reverseRate'),
                               P('Cofact_kcat_rate',0.25, min=0.25e-2, max=0.25e2, xpath='//Reaction[@id="SSHpCof--Cof+SSH"]/forwardRate'),
                               P('actinact_fwd_rate',2.54e-6, min=2.5e-8, max=2.5e-4, xpath='//Reaction[@id="actin+Cof--Cofactin"]/forwardRate'),
                               P('actinact_bckd_rate',0.002, min=0.002e-2, max=0.002e2, xpath='//Reaction[@id="actin+Cof--Cofactin"]/reverseRate'),
                               P('LIMKinact_fwd_rate',0.0005, min=0.0005e-2, max=0.0005e2, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/forwardRate'),
                               P('LIMKinact_bckd_rate',0.04, fixed='LIMKinact_kcat_rate',constant=4, xpath='//Reaction[@id="pLIMK+SSH--SSHpLIMK"]/reverseRate'),
                               P('LIMKinact_kcat_rate',1.15, min=1.15e-2, max=1.15e2, xpath='//Reaction[@id="SSHpLIMK--pLIMK+SSH"]/forwardRate'),
                               P('SSHinact_fwd_rate',3.24e-5, min=3.24e-7, max=3.24e-3, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/forwardRate'),
                               P('SSHinact_bckd_rate',0.13, fixed='SSHinact_kcat_rate',constant=4, xpath='//Reaction[@id="RacPAK+SSH--SSHRacPAK"]/reverseRate'),
                               P('SSHinact_kcat_rate',0.064, min=0.064e-2, max=0.064e2, xpath='//Reaction[@id="SSHRacPAK--pSSH+RacPAK"]/forwardRate'),
                               P('Cofinact_fwd_rate',0.0016, min=0.0016e-2, max=0.0016e2, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/forwardRate'),
                               P('Cofinact_bckd_rate',0.033, fixed='Cofinact_kcat_rate',constant=4, xpath='//Reaction[@id="Cof+pLIMK--pLIMKCof"]/reverseRate'),
                               P('Cofinact_kcat_rate',0.113, min=0.113e-2, max=0.113e2, xpath='//Reaction[@id="pLIMKCof--pCof+pLIMK"]/forwardRate'))

#this command indicates that experiments are from a previous simulation
###################### END CUSTOMIZATION #######################################

fitness = nrd_fitness.specie_concentration_fitness(species_list=mol,start=start_stim)

fit = aju.optimize.Fit(tmpdir, exp, model_set, None, fitness, params,
                       _make_simulation=aju.xml.NeurordSimulation.make,
                       _result_constructor=aju.xml.NeurordResult)
fit.load()
fit.do_fit(iterations, sigma=0.3,popsize=popsize)
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
aju.drawing.plot_history(fit,fit.measurement,mol_dict=mol)

'''
fit[830]                  
NeurordSimulation(<TemporaryDirectory '/tmp/cof_opt/tmprom761ve'>

*************** optimization converged at 0 with m= 0.38459888633036704
*************** optimization converged at 200 with m= 0.3718750411155398
*************** optimization converged at 400 with m= 0.3828367857551935
**************  optimization NOT converged 600 m= 0.33132277813542976
*************** optimization converged at 800 with m= 0.31682237301114896
end of iterate_fit.py cof_opt len of fitness 1000 last_j 5 len(mean_dict) 5
Racact_fwd_rate = 1.2106438509014752e-07 +/- 3.205943287317544e-09
Racact_kcat_rate = 3.191684174483173e-05 +/- 6.733659228341633e-07
PAKact_fwd_rate = 1.2855342315141675e-05 +/- 1.1053498387882872e-07
PAKact_bckd_rate = 0.0018297780934014524 +/- 1.6936387897028285e-05
LIMKact_fwd_rate = 0.0031882398506731225 +/- 5.56061754660724e-05
LIMKact_kcat_rate = 0.058777648045564526 +/- 0.000996389977509354
SSHact_fwd_rate = 2.868778905737035e-06 +/- 6.023113961744283e-08
SSHact_kcat_rate = 9.979647942591331e-05 +/- 1.381976693897793e-06
Cofact_fwd_rate = 4.972222932554047e-06 +/- 1.2260185538710978e-07
Cofact_kcat_rate = 0.3735582010534191 +/- 0.003926755139785789
actinact_fwd_rate = 1.5106484906625724e-08 +/- 5.668342437056704e-11
actinact_bckd_rate = 0.0009392079674340111 +/- 1.1022216210138052e-05
LIMKinact_fwd_rate = 0.0001352746368047536 +/- 1.8159499467142406e-06
LIMKinact_kcat_rate = 0.024214401564152385 +/- 0.00032066194270299436
SSHinact_fwd_rate = 0.00011791766049226193 +/- 1.3860033413869288e-06
SSHinact_kcat_rate = 0.0417416425954855 +/- 0.00024913021721035693
Cofinact_fwd_rate = 0.0010370753216020405 +/- 1.4881874977545987e-05
Cofinact_kcat_rate = 0.05695041546323296 +/- 0.00019545764587265937
'''
