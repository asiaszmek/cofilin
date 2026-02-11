sub_species={'PKA':['PKA'],'PKAr':['PKA','PKAr','PKAcAMP2','PKAcAMP4'],
             'PKAc':['PKAc','KalPKAc','PKAcLIMK','PKAc_PDE4_cAMP','PKAcPDE4','PKAcLR','PKAcpLR','PKAcppLR'],
             'Rac':['pKalRac','RacGap','RacGDP','RacGTP','RacPAK','RacPAKLIMK','RacPAKSSH','RCapGDP'],
             'actCof':['Cof','Cofactin'],'InCof':['pCof'],
             'totPAK':['RacPAK','RacPAKLIMK','RacPAKSSH'],
             'totCofALL':['Cof','Cofactin','pCof','SSHpCof','pLIMKCof'],
             'totCof':['Cof','Cofactin','pCof'],
             'totCofbound':['Cofactin','SSHpCof','pLIMKCof'],
             'totRac':['pKalRac','RacGap','RacGDP','RacGTP','RacPAK','RacPAKLIMK','RacPAKSSH'],
             'CKtot':['KalCKCamCa4','KalCKpCamCa4','CKp','CKCamCa4','CKpCamCa4','CKpPP1','CKpCamCa4PP1'],
             'CKonly':['CKp','CKCamCa4','CKpCamCa4'],
             'epac':['EpacAMP'],
             'LIMK':['RacPAKLIMK','PKAcLIMK','LIMK','pLIMK','pLIMKCof','SSHpLIMK'],
             'ssh':['CaNCamCa4pSSH','pSSH','SSH','SSHpLIMK','SSHpCof','RacPAKSSH','SCappSSH'],
             'Kal':['Kal','KalCKCamCa4','KalCKpCamCa4','KalPKAc','pKal','pKalPP1','pKalRac'],
             'phos':['KalCKpCamCa4','pKal','pKalPP1','pKalRac','CKp','CKCamCa4','CKpCamCa4','CKpPP1','CKpCamCa4PP1','pPDE4','pPDE4cAMP','pLR', 'ppLR', 'ppLRGi'],
             'PKAphos':['pPDE4','pPDE4cAMP','pLR', 'ppLR', 'ppLRGi','EpacAMP'],
             'phoslimk':['KalCKpCamCa4','pKal','pKalPP1','pKalRac','CKp','CKCamCa4','CKpCamCa4','CKpPP1','CKpCamCa4PP1','pPDE4','pPDE4cAMP','pLR', 'ppLR', 'ppLRGi','pLIMK','pLIMKCof','SSHpLIMK'],
             'dualPKA_PAK':{'pLIMK','pLIMKCof','SSHpLIMK'},
             'dualCK_PKA':{'pKal','pKalPP1','pKalRac'}}
##phos=pka and ck, 
tot_species=['totCofbound','totPAK','totRac','totCofALL','totCof']
#tot_species=['CK','PKAc','ncx','pmca','PKAc','Cof','LIMK','SSH','ssh','Rac','Kal','PAK','Cam','CaN','PKAr','actin','Gap','CK','PP1','Ip35','AC1','AC8','PDE4','PDE1','Epac','Calbin']
weight={}
#signature molecules must be in tot_species
signature={}#{'kinase':{'num':['CKtot','PKAphos'],'denom':[]}, 'CK_PKA_PAK':{'num':['CKtot','PKAphos','dualPKA_PAK','dualCK_PKA'],'denom':[]},'CK_PKA':{'num':['CKtot','PKAphos','dualCK_PKA'],'denom':[]},'cof_act':{'num':['actCof'],'denom':[]}}
#thresh keys must be regions in the morphology
thresh={}#{'kinase':{'dend':0.2,'dendsub':0.2,'sa1[0]':0.2},'CK_PKA_PAK':{'dend':0.2,'dendsub':0.2,'sa1[0]':0.3},'CK_PKA':{'dend':0.2,'dendsub':0.2,'sa1[0]':0.3},'cof_act':{'dend':0.2,'dendsub':0.2,'sa1[0]':0.3}}
min_max={}   
'''
min_max = {
    'kinase': {
        'num': {
            'CKtot': {
                'dend': {'max': 15786.59754337646, 'min': 0.8302693915813638},
                'dendsub': {'max': 1692.815505760453, 'min': 0.0},
                'sa1[0]': {'max': 31199.881689536902, 'min': 0.0}
            },
            'PKAphos': {
                'dend': {'max': 1485.2072733874927, 'min': 17.539440897154023},
                'dendsub': {'max': 363.86556086053116, 'min': 10.482151068717961},
                'sa1[0]': {'max': 1700.7886977245707, 'min': 10.482151068717961}
            }
        },
        'denom': {}
    },
    'CK_PKA_PAK': {
        'num': {
            'CKtot': {
                'dend': {'max': 15786.59754337646, 'min': 0.8302693915813638},
                'dendsub': {'max': 1692.815505760453, 'min': 0.0},
                'sa1[0]': {'max': 31199.881689536902, 'min': 0.0}
            },
            'PKAphos': {
                'dend': {'max': 1485.2072733874927, 'min': 17.539440897154023},
                'dendsub': {'max': 363.86556086053116, 'min': 10.482151068717961},
                'sa1[0]': {'max': 1700.7886977245707, 'min': 10.482151068717961}
            },
            'dualPKA_PAK': {
                'dend': {'max': 1609.1186901613787, 'min': 63.72317580386967},
                'dendsub': {'max': 192.62249884687637, 'min': 58.015073736747794},
                'sa1[0]': {'max': 1913.0675878165098, 'min': 58.015073736747794}
            },
            'dualCK_PKA': {
                'dend': {'max': 370.7907623766766, 'min': 0.0},
                'dendsub': {'max': 244.61811949465928, 'min': 0.0},
                'sa1[0]': {'max': 6337.675778889454, 'min': 0.0}
            }
        },
        'denom': {}
    },
    'CK_PKA': {
        'num': {
            'CKtot': {
                'dend': {'max': 15786.59754337646, 'min': 0.8302693915813638},
                'dendsub': {'max': 1692.815505760453, 'min': 0.0},
                'sa1[0]': {'max': 31199.881689536902, 'min': 0.0}
            },
            'PKAphos': {
                'dend': {'max': 1485.2072733874927, 'min': 17.539440897154023},
                'dendsub': {'max': 363.86556086053116, 'min': 10.482151068717961},
                'sa1[0]': {'max': 1700.7886977245707, 'min': 10.482151068717961}
            },
            'dualCK_PKA': {
                'dend': {'max': 370.7907623766766, 'min': 0.0},
                'dendsub': {'max': 244.61811949465928, 'min': 0.0},
                'sa1[0]': {'max': 6337.675778889454, 'min': 0.0}
            }
        },
        'denom': {}
    },
    'cof_act': {
        'num': {
            'actCof': {
                'dend': {'max': 1009.8465971089979, 'min': 5.604318393174205},
                'dendsub': {'max': 119.24744136587337, 'min': 5.604318393174205},
                'sa1[0]': {'max': 2117.67374694127, 'min': 3.2172938923777847}
            }
        },
        'denom': {}
    }
}


 #{'cof':{'num':{'max':1497.9, 'min':1.45},'denom':{'max':1929, 'min':37}}}
'''
