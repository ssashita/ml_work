#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelBinarizer,LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import lil_matrix,csr_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy import stats as ss
import matplotlib.pyplot as plt

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr_names):
        self.attribute_names=attr_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class MyLabelFillNA(TransformerMixin):
    def __init__(self,fill_with="unknown", *args, **kwargs):
        self.fill_with = fill_with
    def fit(self, x,y=0):
        return self
    def transform(self, x, y=0):
        retval=None
        if isinstance(x,pd.DataFrame):
            retval = x.fillna(self.fill_with)
        elif isinstance(x, np.ndarray):
            retval = pd.DataFrame(x).fillna(self.fill_with)
        else:
            raise Exception("input arg needs to be pandas DataFrame or numpy array")
        return retval.values

class MyLabelEncoder(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

class MyMultiLabelEncoder(TransformerMixin):
    def __init__(self, label_encoder_args_array=None ):
        def f( i):
            if label_encoder_args_array==None or label_encoder_args_array[i] ==None: return MyLabelEncoder()
            else: return MyLabelBinarizer(*label_encoder_args_array[i])
        self.label_encoder_args_array= label_encoder_args_array
        self.encoders=None
        if label_encoder_args_array is not  None:
            self.encoders = [f(i) for i in range(len(label_encoder_args_array))]
            
    def fit(self,x,y=0):
        xt = x.transpose()
        if self.encoders==None:
            self.encoders = [MyLabelEncoder() for i in range(len(xt))]
        print(xt.shape,len(xt),len(self.encoders))
        for i in range(len(xt)):
            arr=xt[i]
            enc=self.encoders[i]
            #y=arr.reshape(-1,1)
            enc.fit(arr)
        return self
    
    def transform(self,x,y=0):
        xx=None
        xt=x.transpose()
        for i in range(len(xt)):
            enc = self.encoders[i]
            arr= xt[i]
            #y=arr.reshape(-1,1)
            z=enc.transform(arr).reshape(-1,1)
            if i==0:
                xx=z
            else:
                xx=np.concatenate((xx,z),axis=1)
        print('xx shape is',xx.shape)
        return lil_matrix(xx)
        
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

class MyMultiLabelBinarizer(TransformerMixin):
    
    def __init__(self, binarizer_args_array=None ):
        def f( i):
            if binarizer_args_array==None or binarizer_args_array[i] ==None: return MyLabelBinarizer()
            else: return MyLabelBinarizer(*binarizer_args_array[i])
        self.binarizer_args_array= binarizer_args_array
        self.encoders=None
        if binarizer_args_array is not  None:
            self.encoders = [f(i) for i in range(len(binarizer_args_array))]
    def fit(self,x,y=0):
        xt = x.transpose()
        if self.encoders==None:
            self.encoders = [MyLabelBinarizer() for i in range(len(xt))]
        print(xt.shape,len(xt),len(self.encoders))
        for i in range(len(xt)):
            arr=xt[i]
            enc=self.encoders[i]
            y=arr.reshape(-1,1)
            enc.fit(y)
        return self
    
    def transform(self,x,y=0):
        xx=None
        xt=x.transpose()
        for i in range(len(xt)):
            enc = self.encoders[i]
            arr= xt[i]
            y=arr.reshape(-1,1)
            z=enc.transform(y)
            if i==0:
                xx=z
            else:
                xx=np.concatenate((xx,z),axis=1)
        print('xx shape is',xx.shape)
        return lil_matrix(xx)
        
class FullPipeline:

    def full_pipeline_apply_features(self,data, non_num_attrs=None, num_attrs=None):
        num_pipeline=None
        full_pipeline=None
        if num_attrs != None:
            num_pipeline = Pipeline([('num_selector', DataFrameSelector(num_attrs)),('imputer',SimpleImputer(strategy='median')), ('std_scaler',StandardScaler() )])
            full_pipeline= num_pipeline
            print('numattrs is not None')

        cat_pipeline=None
        if non_num_attrs != None:
            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(non_num_attrs)),
                ('na_filler', MyLabelFillNA("Unknown")),
                ('label_encoder', MyMultiLabelBinarizer())
            ])
            full_pipeline=cat_pipeline


        #num_pipeline.fit_transform(data)
        #cat_pipeline.fit_transform(data)
        #MyLabelBinarizer().fit_transform(selected_data)
        if num_pipeline != None and cat_pipeline != None:
            print('Both num_pipeline and cat_pipeline exist')
            full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
            ])
        if full_pipeline != None:
            self.full_features_pipeline_=full_pipeline
            return full_pipeline.fit_transform(data)
        return None

    def full_pipeline_apply_labels(self,data, label_data_non_num):
        label_binarized_pipeline = Pipeline([('selector', DataFrameSelector(list(label_data_non_num))),
        ('na_filler', MyLabelFillNA("Unknown")),
        ('label_encoder', MyLabelBinarizer())])
        label_binarized_data_prepared = label_binarized_pipeline.fit_transform(data)
        self.label_pipeline_ = label_binarized_pipeline
        return label_binarized_data_prepared
    
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def conditional_probabilities(data,xattr,yattr):
    d=data[[xattr,yattr]]
    dg=d.groupby(yattr)
    return dg.value_counts()/dg.count()

def plot_precision_recall_vs_threshold(precisions, recalls,thresholds):
    plt.plot(thresholds, precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds,recalls[:-1], "g-",label="Recall")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr,linewidth=2, label=label) #tpr is the recall or true positives rate
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[ ]:




