sub_species={'PKA':['PKA'],'PKAr':['PKA','PKAr','PKAcAMP2','PKAcAMP4'],
             'PKActot':['PKAc','KalPKAc','PKAcLIMK','PKAc_PDE4_cAMP','PKAcPDE4','PKAcLR','PKAcpLR','PKAcppLR'],
             'Rac':['pKalRac','RacGap','RacGDP','RacGTP','RacPAK','RacPAKLIMK','RacPAKSSH'],
             'totRac':['pKalRac','RacGap','RacGDP','RacGTP','RacPAK','RacPAKLIMK','RacPAKSSH','RCapGDP'],
             'actCof':['Cof','Cofactin'],'InCof':['pCof'],
             'totPAK':['PAK','RacPAK','RacPAKLIMK','RacPAKSSH'],
             'RacPAK':['RacPAK','RacPAKLIMK','RacPAKSSH'],
             'totCofALL':['Cof','Cofactin','pCof','pLIMKCof','SSHpCof'],
             'totCof':['Cof','Cofactin','pCof'],
             'pCoftot':['pCof','SSHpCof'],
             'CKtot':['KalCKCamCa4','KalCKpCamCa4','CKp','CKCamCa4','CKpCamCa4','CKpPP1','CKpCamCa4PP1'],
             'CKonly':['CKp','CKCamCa4','CKpCamCa4'],
             'LIMKtot':['RacPAKLIMK','PKAcLIMK','LIMK','pLIMK','pLIMKCof','SSHpLIMK'],
             'LIMKall':['RacPAKLIMK','PKAcLIMK','LIMK'],
             'pLIMKall':['pLIMK','pLIMKCof','SSHpLIMK'],
             'ssh':['CaNCamCa4pSSH','pSSH','SSH','SSHpLIMK','SSHpCof','RacPAKSSH','SCappSSH'],
             'Kal':['Kal','KalCKCamCa4','KalCKpCamCa4','KalPKAc','pKal','pKalPP1','pKalRac'],
             'phos':['KalCKpCamCa4','pKal','pKalPP1','pKalRac','CKp','CKCamCa4','CKpCamCa4','CKpPP1','CKpCamCa4PP1','pPDE4','pPDE4cAMP','pLR', 'ppLR', 'ppLRGi'],
             'PKAphos':['pPDE4','pPDE4cAMP','pLR', 'ppLR', 'ppLRGi'],#'EpacAMP'],
             'phoslimk':['KalCKpCamCa4','pKal','pKalPP1','pKalRac','CKp','CKCamCa4','CKpCamCa4','CKpPP1','CKpCamCa4PP1','pPDE4','pPDE4cAMP','pLR', 'ppLR', 'ppLRGi','pLIMK','pLIMKCof','SSHpLIMK'],
             'dualPKA_PAK':['pLIMK','pLIMKCof','SSHpLIMK'],
             'dualCK_PKA':['pKal','pKalPP1','pKalRac'],
             'Gs':['GsR', 'LRGs','Gs', 'Gsbg'],
             'R':['R','LR','GsR','LRGs','pLR'],
             'GasGTP':['GasGDP', 'GasGTP','AC1GasGTP','AC1GasGTPCamCa4','AC1GasGTPCamCa4ATP'],
             'PP1':['PP1','Ip35PP1', 'pKalPP1','CKpPP1','CKpCamCa4PP1'],
             'I1':['Ip35','Ip35PP1','Ip35CaNCamCa4','Ip35PP1CaNCamCa4','I1']}
##phos=pka and ck, 
tot_species=['dualPKA_PAK','PKAphos','dualCK_PKA','CKtot','actCof']# 'totCofALL','totCof']#, 'pCoftot','pLIMKall','LIMKall','totCofALL','totCof','RacPAK']# 'PKActot',
#tot_species=['CK','totCofALL','LIMKtot','ssh','totRac','Kal','totPAK','CaN','Ng','PKAr','PKActot','actin','Gap','CK','PP1','I1','PDE4','Epac','Calbin', 'AC8', 'AC1', 'Gs','R','GasGTP','RCap','SCap'] #sum molecules to ensure totals are correct
#tot_species=['pCoftot','pLIMKall','LIMKall']
weight={}
#signature molecules must be in tot_species or in molecules
signature={'kinase':{'num':['CKtot','PKAphos','EpacAMP'],'denom':[]},
           'CK_PKA_PAK':{'num':['CKtot','PKAphos','dualPKA_PAK','dualCK_PKA'],'denom':[]},
           'cof_act':{'num':['actCof'],'denom':[]},
           'CK_PKA_Gi':{'num':['CKtot','PKAphos','dualCK_PKA','EpacAMP','Gibg'],'denom':[]}}
#thresh keys must be regions in the morphology.  Only influences plotting, not output
#add in sa1[1] and sa1[2] for 10um dendrite
thresh={'kinase':{'dend':0.1,'sa1[0]':0.1,'sa1[1]':0.1,'sa1[2]':0.1},
        'CK_PKA_PAK':{'dend':0.1,'sa1[0]':0.1,'sa1[1]':0.1,'sa1[2]':0.1},
        'cof_act':{'dend':0.1,'sa1[0]':0.1,'sa1[1]':0.1,'sa1[2]':0.1},
        'CK_PKA_Gi':{'dend':0.1,'sa1[0]':0.1,'sa1[1]':0.1,'sa1[2]':0.1}}

#signature={}
#thresh={}
min_max={}   
min_max = {
    'kinase': {
        'num': {
            'CKtot': {
                'dend': {'max': 9935.007, 'min': 8.809},
                'sa1[0]': {'max': 35541.369, 'min': 0.0},
                'sa1[1]': {'max': 35541.369, 'min': 0.0},
                'sa1[2]': {'max': 35541.369, 'min': 0.0}
                 },
            'PKAphos': {
                'dend': {'max': 1731.065, 'min': 92.499},
                'sa1[0]': {'max': 1858.08, 'min': 83.69},
                'sa1[1]': {'max': 1858.08, 'min': 83.69},
                'sa1[2]': {'max': 1858.08, 'min': 83.69}
                 },
            'EpacAMP': {
                'dend': {'max': 342.486, 'min': 0.704},
                'sa1[0]': {'max': 419.443, 'min': 0.0},
                'sa1[1]': {'max': 419.443, 'min': 0.0},
                'sa1[2]': {'max': 419.443, 'min': 0.0},
                 }
            },
        'denom': {}
            },
    'CK_PKA_PAK': {
        'num': {
            'dualPKA_PAK': {
                'dend': {'max': 1588.676, 'min': 70.365},
                'sa1[0]': {'max': 1704.625, 'min': 52.526},
                'sa1[1]': {'max': 1704.625, 'min': 52.526},
                'sa1[2]': {'max': 1704.625, 'min': 52.526},
                 },
            'dualCK_PKA': {
                'dend': {'max': 346.024, 'min': 0.0},
                'sa1[0]': {'max': 6189.336, 'min': 0.0},
                'sa1[1]': {'max': 6189.336, 'min': 0.0},
                'sa1[2]': {'max': 6189.336, 'min': 0.0},
            },
            'CKtot': {
                'dend': {'max': 9935.007, 'min': 8.809},
                'sa1[0]': {'max': 35541.369, 'min': 0.0},
                'sa1[1]': {'max': 35541.369, 'min': 0.0},
                'sa1[2]': {'max': 35541.369, 'min': 0.0}
            },
            'PKAphos': {
                'dend': {'max': 1731.065, 'min': 92.499},
                'sa1[0]': {'max': 1858.08, 'min': 83.69},
                'sa1[1]': {'max': 1858.08, 'min': 83.69},
                'sa1[2]': {'max': 1858.08, 'min': 83.69},
            },
            'EpacAMP': {
                'dend': {'max': 342.486, 'min': 0.704},
                'sa1[0]': {'max': 419.443, 'min': 0.0},
                'sa1[1]': {'max': 419.443, 'min': 0.0},
                'sa1[2]': {'max': 419.443, 'min': 0.0},
            }
        },
        'denom': {}
    },
    'CK_PKA_Gi': {
        'num': {
            'dualCK_PKA': {
                'dend': {'max': 346.024, 'min': 0.0},
                'sa1[0]': {'max': 6189.336, 'min': 0.0},
                'sa1[1]': {'max': 6189.336, 'min': 0.0},
                'sa1[2]': {'max': 6189.336, 'min': 0.0},
            },
            'CKtot': {
                'dend': {'max': 9935.007, 'min': 8.809},
                'sa1[0]': {'max': 35541.369, 'min': 0.0},
                'sa1[1]': {'max': 35541.369, 'min': 0.0},
                'sa1[2]': {'max': 35541.369, 'min': 0.0}
            },
            'PKAphos': {
                'dend': {'max': 1731.065, 'min': 92.499},
                'sa1[0]': {'max': 1858.08, 'min': 83.69},
                'sa1[1]': {'max': 1858.08, 'min': 83.69},
                'sa1[2]': {'max': 1858.08, 'min': 83.69},
            },
            'EpacAMP': {
                'dend': {'max': 342.486, 'min': 0.704},
                'sa1[0]': {'max': 419.443, 'min': 0.0},
                'sa1[1]': {'max': 419.443, 'min': 0.0},
                'sa1[2]': {'max': 419.443, 'min': 0.0},
            },
            'Gibg': {
                'dend': {'max': 512.471, 'min': 0.0},
                'sa1[0]': {'max': 195.655, 'min': 0.0},
                'sa1[1]': {'max': 195.655, 'min': 0.0},
                'sa1[2]': {'max': 195.655, 'min': 0.0},
            }
        },
        'denom': {}
    },
    'cof_act': {
        'num': {
            'actCof': {
                'dend': {'max': 1108.441, 'min': 101.308},
                'sa1[0]': {'max': 2452.716, 'min': 79.285},
                'sa1[1]': {'max': 2452.716, 'min': 79.285},
                'sa1[2]': {'max': 2452.716, 'min': 79.285},
            }
        },
        'denom': {}
    }
}


if __name__ == '__main__':
    fail=False
    for key in signature.keys():
        if key in min_max:
            for nd in ['num','denom']:
                for mol in signature[key][nd]:
                    if not mol in min_max[key][nd]:
                        print('key=',key,', mol',mol,'in sig, not in min_max[key][',nd,']')
                        fail=True
        else:
            print('key=',key,'in signature, not in min_max - norm signature will fail')
            fail=True
        if key not in thresh.keys():
            print('key=',key,'in signature dict, missing from thresh dictionary')
            fail=True
    for key in thresh.keys():
        if key not in signature.keys():
            print('key=',key,'in thresh dict, missing from sig - plot_signature will fail')
            fail=True
    if fail==False:
        print('signature and thresh are well defined')
