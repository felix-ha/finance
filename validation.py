import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import pickle
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import calibration_curve



def plot_hist(df_train, df_valid):
    fig = plt.figure(figsize=(12, 10))

    # Histogram
    plt.subplot(1, 2, 1)
    df_0 = df_train[df_train['y_true'] == 0]
    df_1 = df_train[df_train['y_true'] == 1]
    
    sns.distplot(df_0['y_prob'] , color="red", label="y_prob 0", bins=np.arange(0,1.1,0.05))
    ax = sns.distplot(df_1['y_prob'] , color="blue", label="y_prob 1", bins=np.arange(0,1.1,0.05))
    
  
    plt.title('training ', fontsize=10)
    plt.legend(loc="upper left")
    
   
    
    # Histogram
    plt.subplot(1, 2, 2)
    df_0 = df_valid[df_valid['y_true'] == 0]
    df_1 = df_valid[df_valid['y_true'] == 1]
    
    sns.distplot(df_0['y_prob'] , color="red", label="y_prob 0", bins=np.arange(0,1.1,0.05))
    ax = sns.distplot(df_1['y_prob'] , color="blue", label="y_prob 1", bins=np.arange(0,1.1,0.05))
    
    plt.title('validation ', fontsize=10)

    plt.legend(loc="upper left")
    
    
    
    #fig.title('title', fontsize=10)
    fig.savefig('plots/stat_summary.png', dpi=300, bbox_inches='tight')
    
    

def bootstrap_auc_ci(y_true, y_prob, n_bootstrap_samples):

    aucs = np.empty(n_bootstrap_samples)
    
    for i in range(n_bootstrap_samples):
       
        ids = np.random.randint(low=0, high=len(y_true), size=len(y_true))
        
        y_true_samples = y_true[ids]
        y_prob_samples = y_prob[ids]
        
        aucs[i] = get_auc(y_true_samples, y_prob_samples) 
    
    
    
    aucs = np.array(aucs)
    aucs = aucs[~np.isnan(aucs)]
    
    #aucs_mean = np.mean(aucs)
    #aucs = np.sort(aucs)
    
    ci_bottom, ci_top = np.percentile(aucs, [2.5, 97.5])
    
    return ci_bottom, ci_top


def get_auc(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return auc(fpr, tpr)


def plot_box(df_result, name, n_bootstrap_samples):
    y_true = df_result['y_true']
    y_prob = df_result['y_prob']
    
    y_pred = np.where(y_prob > 0.5, 1, 0)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    auroc = get_auc(y_true, y_prob)
    
    ci_bottom, ci_top =  bootstrap_auc_ci(y_true, y_prob, n_bootstrap_samples)

    
    #accuracy_benchmark = np.mean(y_true)
    
    
    # Boxplot
    ax = sns.catplot(x="y_true", y="y_prob", kind="box", data=df_result)
    ax = sns.stripplot(x="y_true", y="y_prob",jitter=0.1, alpha=0.5, data=df_result)
    fig = ax.get_figure()
    plt.title('AUC: {:.3f} ( {:.3f} |  {:.3f} )'.format( auroc, ci_bottom, ci_top), fontsize=10)
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
    
    
def plot_losses(training_losses, valid_losses):
    fig = plt.figure(figsize=(10, 10))
    
    x = range(len(training_losses))
    
    plt.plot(x, training_losses, 'b', label='training')
    plt.plot(x, valid_losses, 'r', label='validation')
    plt.title('Training summary')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc="upper left")
    
    fig.savefig('plots/training_summary.png', bbox_inches='tight')
    
    
def run_validation(n_bins, n_bootstrap_samples):
    with open(r'temp\result.pkl', 'rb') as pick:
        data_dict = pickle.load(pick)
              
    
    df_result_train = data_dict['df_result_train']
    y_true_train = df_result_train['y_true']
    y_prob_train = df_result_train['y_prob']
    name_train = 'Train'
    
    df_result_val = data_dict['df_result_val']
    y_true_val = df_result_val['y_true']
    y_prob_val = df_result_val['y_prob']
    name_val = 'Validation'
    

    
    plot_hist(df_result_train, df_result_val)
    plot_box(df_result_train, name_train, n_bootstrap_samples)
    plot_box(df_result_val, name_val, n_bootstrap_samples)
    calibration_plot(y_true_train, y_prob_train, y_true_val, y_prob_val, n_bins=n_bins) 
    
    try:
        training_losses = data_dict['training_losses']
        valid_losses = data_dict['valid_losses']
        plot_losses(training_losses, valid_losses)
    except:
        print("no losses found, no plot for losses created")

if __name__ == '__main__':
    run_validation(10, n_bootstrap_samples=1000)


#from matplotlib import pyplot as plt
#plt.style.use('ggplot')#

#x = range(training_epochs)

#plt.subplot(2, 1, 1)
#plt.plot(x, losses, 'b')
#plt.title('Training summary')
#plt.ylabel('loss')

#plt.subplot(2, 1, 2)
#plt.plot(x, test_acc, 'b')
#plt.xlabel('epochs')
#plt.ylabel('training accuracy')

#plt.savefig('training_summary.png', dpi=500)