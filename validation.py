import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import calibration_curve


def plot_hist_box(df_result, name):
    y_true = df_result['y_true']
    y_prob = df_result['y_prob']
    
    y_pred = np.where(y_prob > 0.5, 1, 0)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)
    
    accuracy_benchmark = np.mean(y_true)
    
    #print("\nBenchmark accuray   : {:.5f}".format(accuracy_benchmark))
    #print("Validation accuray  : {:.5f}".format(accuracy))
    #print("Validation auc    : {:.5f}\n".format(auroc))
    
    
    
    # Histogram
    df_0 = df_result[df_result['y_true'] == 0]
    df_1 = df_result[df_result['y_true'] == 1]
    
    sns.distplot(df_0['y_prob'] , color="red", label="y_prob 0", bins=np.arange(0,1.1,0.01))
    ax = sns.distplot(df_1['y_prob'] , color="blue", label="y_prob 1", bins=np.arange(0,1.1,0.01))
    
    fig = ax.get_figure()
    #plt.title('Title ', fontsize=10)
    fig.suptitle(name, fontsize=10)
    plt.legend()
    plt.show()
    fig.savefig('plots/histogram_' + name +'.png', dpi=300, bbox_inches='tight')
    
    
    # Boxplot
    ax = sns.catplot(x="y_true", y="y_prob", kind="box", data=df_result)
    ax = sns.stripplot(x="y_true", y="y_prob",jitter=0.1, alpha=0.5, data=df_result)
    fig = ax.get_figure()
    plt.title('Benchmark ACC: {:.3f} | ACC {:.3f}  | AUC: {:.3f}'.format( accuracy_benchmark, accuracy, auroc), fontsize=10)
    fig.suptitle(name, fontsize=8)
    plt.show()
    fig.savefig('plots/categorial_' + name +'.png', dpi=300, bbox_inches='tight')


def calibration_plot(y_true_train, y_prob_train, y_true_val, y_prob_val, n_bins):
    fig = plt.figure(figsize=(10, 10))
    
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    
    #for i in range(len(y_prob_best_clf_train)):
    #    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)
    #    name = results.iloc[i]['Model']
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true_train, y_prob_train, n_bins=n_bins)
    name = 'Train'
        
    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))
    
    
    
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true_val, y_prob_val, n_bins=n_bins)
    name = 'Validation'
        
    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))
    
    
    
    
    
    plt.xlim(-0.05, 1.05)
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title('calibration plot')
    plt.legend(loc="upper left")
    fig.savefig('plots/calib_train.png', bbox_inches='tight')
    
    
def run_validation(n_bins):
    df_result_train = pd.read_pickle('temp/result_train.pkl')
    y_true_train = df_result_train['y_true']
    y_prob_train = df_result_train['y_prob']
    name_train = 'Train'
    
    df_result_val = pd.read_pickle('temp/result_val.pkl')
    y_true_val = df_result_val['y_true']
    y_prob_val = df_result_val['y_prob']
    name_val = 'Validation'
    
    plot_hist_box(df_result_train, name_train)
    plot_hist_box(df_result_val, name_val)
    calibration_plot(y_true_train, y_prob_train, y_true_val, y_prob_val, n_bins=n_bins)    

if __name__ == '__main__':
    run_validation(10)

