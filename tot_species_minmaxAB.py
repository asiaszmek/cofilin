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
tot_species=[ 'dualPKA_PAK','PKAphos','dualCK_PKA','CKtot','actCof']#, 'pCoftot','pLIMKall','LIMKall','totCofALL','totCof','RacPAK']# 'PKActot',
#tot_species=['CK','totCofALL','LIMKtot','ssh','totRac','Kal','totPAK','CaN','Ng','PKAr','PKActot','actin','Gap','CK','PP1','I1','PDE4','Epac','Calbin', 'AC8', 'AC1', 'Gs','R','GasGTP','RCap','SCap'] #sum molecules to ensure totals are correct
#tot_species=['pCoftot','pLIMKall','LIMKall']
weight={}
#signature molecules must be in tot_species or in molecules
signature={'kinase':{'num':['CKtot','PKAphos','EpacAMP'],'denom':[]}, 'CK_PKA_PAK':{'num':['CKtot','PKAphos','dualPKA_PAK','dualCK_PKA'],'denom':[]},'cof_act':{'num':['actCof'],'denom':[]}}#,'CK_PKA':{'num':['CKtot','PKAphos','dualCK_PKA','EpacAMP'],'denom':[]}}
#thresh keys must be regions in the morphology
thresh={'kinase':{'dend':0.2,'dendsub':0.25,'sa1[0]':0.25},'CK_PKA_PAK':{'dend':0.2,'dendsub':0.2,'sa1[0]':0.2},'cof_act':{'dend':0.1,'dendsub':0.2,'sa1[0]':0.15}}#,'CK_PKA':{'dend':0.2,'dendsub':0.2,'sa1[0]':0.25}}
#signature={}
#thresh={}
min_max={}   
min_max = {
    'kinase': {
        'num': {
            'CKtot': {
                'dend': {'max': 9940.982, 'min': 1.557},
                'dendsub': {'max': 1121.59, 'min': 0.0},
                'sa1[0]': {'max': 35419.884, 'min': 0.0}
            },
            'PKAphos': {
                'dend': {'max': 1714.868, 'min': 21.172},
                'dendsub': {'max': 382.91, 'min': 21.172},
                'sa1[0]': {'max': 1916.904, 'min': 21.172}
            },
            'EpacAMP': {
                'dend': {'max': 321.808, 'min': 0.0},
                'dendsub': {'max': 344.043, 'min': 0.0},
                'sa1[0]': {'max': 350.388, 'min': 0.0}
            }
        },
        'denom': {}
    },
    'CK_PKA_PAK': {
        'num': {
            'dualPKA_PAK': {
                'dend': {'max': 1681.61, 'min': 65.28},
                'dendsub': {'max': 194.024, 'min': 57.133},
                'sa1[0]': {'max': 1777.516, 'min': 57.133}
            },
            'dualCK_PKA': {
                'dend': {'max': 449.965, 'min': 0.0},
                'dendsub': {'max': 297.34, 'min': 0.0},
                'sa1[0]': {'max': 7048.682, 'min': 0.0}
            },
            'CKtot': {
                'dend': {'max': 9940.982, 'min': 1.557},
                'dendsub': {'max': 1121.59, 'min': 0.0},
                'sa1[0]': {'max': 35419.884, 'min': 0.0}
            },
            'PKAphos': {
                'dend': {'max': 1714.868, 'min': 21.172},
                'dendsub': {'max': 382.91, 'min': 21.172},
                'sa1[0]': {'max': 1916.904, 'min': 21.172}
            },
            'EpacAMP': {
                'dend': {'max': 321.808, 'min': 0.0},
                'dendsub': {'max': 344.043, 'min': 0.0},
                'sa1[0]': {'max': 350.388, 'min': 0.0}
            }
        },
        'denom': {}
    },
    'cof_act': {
        'num': {
            'actCof': {
                'dend': {'max': 952.766, 'min': 18.629},
                'dendsub': {'max': 115.407, 'min': 12.869},
                'sa1[0]': {'max': 2283.916, 'min': 12.869}
            }
        },
        'denom': {}
    }
}


if __name__ == '__main__':
    for key in signature.keys():
        if key in min_max:
            for nd in ['num','denom']:
                for mol in signature[key][nd]:
                    if not mol in min_max[key][nd]:
                        print('key=',key,', mol',mol,'in sig, not in min_max[key][',nd,']')
        else:
            print('key=',key,'in signature, not in min_max - norm signature will fail')
        if key not in thresh.keys():
            print('key=',key,'in signature dict, missing from thresh dictionary')
    for key in thresh.keys():
        if key not in signature.keys():
            print('key=',key,'in thresh dict, missing from sig - plot_signature will fail')
