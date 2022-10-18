#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd 
import numpy as np
import jax.numpy as jnp

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  LabelEncoder
from sklearn.compose import ColumnTransformer

path = "/Users/dong/Documents/Project/DualVAE"
os.chdir(path)


def prostate_processing():

    #read revised gene info data
    GeneSymbol = pd.read_csv("/Users/dong/Documents/Project/DualVAE/data/mayo/mayo_geneinfo.csv",header=0)

    #gene expression
    df_ge = pd.read_csv("/Users/dong/Documents/Project/DualVAE/data/mayo/normalizedcounts.eqtl.tsv",delimiter='\t',header=0)
    df_ge = df_ge.rename(columns={"GeneID":"symbol"})
    #filter genes that have a name
    df_ge = df_ge.filter(items=GeneSymbol["X1"],axis=0)
    
    df_ge = df_ge.drop(labels = "symbol",axis = 1).T.sort_index()
    #drop outliers
    df_ge = df_ge.drop("1149")
    df_ge = df_ge.astype("float")


    df_gt = pd.read_csv("/Users/dong/Documents/Project/DualVAE/data/mayo/PC_PRS_269.10.14.19prostate.raw",delimiter='\t',header=0)
    df_gt = df_gt.drop(labels = ['FID','PAT','MAT','SEX','PHENOTYPE'],axis = 1)
    df_gt = df_gt.set_index(df_gt['IID']).drop(labels = "IID",axis = 1).sort_index()
    df_gt = df_gt.drop(1149)
    #drop the feature with same value across the data
    nunique = df_gt.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df_gt = df_gt.drop(cols_to_drop,axis = 1)
    #save snp info
    df_genotype = df_gt.columns.tolist() 
    #modify SNP name
    sep = '_'
    stripped = [x.split(sep,1)[0] for x in df_genotype]
    df_stripped = pd.DataFrame(stripped,columns=["snps"])
    #modify some SNP name
    df_stripped[df_stripped["snps"] == "6:134292717:GTGTTGT:G"] = "rs71681861"
    df_stripped[df_stripped["snps"] == "8:128074815:A:T"] = "rs72725854"
    df_stripped[df_stripped["snps"] == "rs59222162"] = "rs577952184"
    df_stripped[df_stripped["snps"] == "rs74273823"] = "rs141811748"
    df_stripped[df_stripped["snps"] == "rs66883347"] = "rs148511027"
    df_stripped[df_stripped["snps"] == "rs397698226"] = "rs533722308"
    
    
    #cov data, no NAs
    df_cov = pd.read_csv("/Users/dong/Documents/Project/DualVAE/data/mayo/covariates.eqtl.tsv",delimiter='\t',header=0)
    df_cov = df_cov.set_index(df_cov['IID']).drop(labels = "IID",axis = 1).sort_index()
    df_cov_dummy = pd.get_dummies(df_cov,columns = ['Group'],drop_first=True)
    df_cov_dummy = df_cov_dummy.drop(1149)
    df_removePC = df_cov_dummy[["epithelium","tils","Group_low gleason"]]


    #pc for GT
    df_gt_pc = pd.read_csv("/Users/dong/Documents/Project/DualVAE/data/mayo/eigenvec",delimiter='\t',header=0)
    df_gt_pc = df_gt_pc.set_index(df_gt_pc['#IID']).drop(labels = "#IID",axis = 1).sort_index()
    
    def data_stdz(data):
        x = np.asarray(data)
        x -= jnp.mean(x, axis=0)
        x /= jnp.std(x, axis=0)
        return(x)
    
    #standardize data
    data_gt = data_stdz(df_gt) 
    data_ge = data_stdz(df_ge)
    
    #function to run linear regression
    #input dataset, specify covariates and the output form
    def linear_res(X,Y,d_type = "df"):
        #run linear model and calculate residuals
        lm = LinearRegression().fit(X,Y)
        lm_pred = lm.predict(X)
        lm_res = Y - lm_pred
        #output residuals
        if d_type == "df":
            return(lm_res)
        elif d_type == "array":
            return(np.asarray(lm_res))

    data_gt_res = linear_res(df_gt_pc,data_gt,d_type = "array")
    data_ge_res = linear_res(df_cov_dummy,data_ge,d_type = "array")
    #data_ge_res = linear_res(df_removePC,data_ge,d_type = "array")
  
    return GeneSymbol,df_stripped,data_ge_res,data_gt_res,df_gt_pc,df_cov_dummy


def gtex_zscore():
    gtex = pd.read_csv("/Users/dong/Documents/Project/SuSiE/SusiEPCA/data/gtexEQTL_zscore.csv",header=0,index_col=0)
    linear_res = np.loadtxt("/Users/dong/Documents/Project/SuSiE/SusiEPCA/data/gtex_zscore_res.txt")
    susie_res = np.loadtxt("/Users/dong/Documents/Project/SuSiE/SusiEPCA/data/resid.txt")
    pc_res = np.loadtxt("/Users/dong/Documents/Project/SuSiE/SusiEPCA/data/pc_resid.txt")
    pc_res_1 = np.loadtxt("/Users/dong/Documents/Project/SuSiE/SusiEPCA/data/pc_resid_1.txt")
    
    import pickle
    with open("/Users/dong/Documents/Project/SuSiE/SusiEPCA/data/gtex_gene_symbol", "rb") as fp:   # Unpickling
        gene_symbol = pickle.load(fp)
    
    tissue_name = gtex.columns.tolist()
    gtex_array = jnp.asarray(gtex)
    def data_stdz(data):
        x = np.asarray(data.T)
        x -= jnp.mean(x, axis=0)
        x /= jnp.std(x, axis=0)
        return(x.T)
    #gtex_array = data_stdz(gtex_array)
    return gtex_array,linear_res,susie_res,pc_res,pc_res_1,gene_symbol,tissue_name

