import numpy as np

def classify0(test, features, y):
    #Calculate Probability then
    counter=np.zeros((1,22));
    counter_feature=np.zeros((1,22));
    counter_feature_uncondition = np.zeros((1,22));
    for j in range(features[0].size):
        for i in range(features.shape[0]):
            if y[i]==1:#A==1 condition label==1
                counter[0][j] = counter[0][j]+1;
                if features[i,j]== test[j]: #conditional counter for the condition when label==1 P(B|A==1)
                    counter_feature[0][j]=counter_feature[0][j]+1;
            if features[i, j] == test[j]:#unconditional  counter for all test[j] P(B)
                counter_feature_uncondition[0][j] = counter_feature_uncondition[0][j]+1;

    p_poision= counter / features.shape[0]; #P(A==1)
    p_safe = 1-p_poision; #P(A==0)
    P_features_lighton=counter_feature/counter;#P(B|A==1)
    P_features= counter_feature_uncondition / features.shape[0];#P(B)
    #counter of P(B)-P(B|A)=counter of P(B|A==0)/counter of P(A==0) get P(B|A==0)
    P_features_lightoff = (counter_feature_uncondition-counter_feature) / (features.shape[0] - counter);
    #I use ln function in numpy to make sure predicted labels seperated better
    p_poison_test = np.sum(np.log(p_poision) +np.log(P_features_lighton)- np.log(P_features));
    p_safe_test = np.sum(np.log(p_safe) + np.log(P_features_lightoff) - np.log(P_features));
    if p_poison_test>=p_safe_test:
        return 1
    else:
        return 0
