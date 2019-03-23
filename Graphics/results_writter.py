import numpy as np
import matplotlib 
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
from matplotlib import path, rcParams
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime

class Results_writter():
    """
    Class Results_writter:
    - Parameters:
        model_name: Vector containing the image paths
        config: Parameters to reproduce experiment
        results: obtained_results
        predicts: model predictionx
        labels: original labels

    """
    def __init__(self, model_name, config, results):
        self.model_name = model_name
        self.config = config
        self.results = results
        self.class_names = ['AnyConnect', 'Interflex', 'Word', 'Outlook', 'Skype']
        self.data_actual = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def save(self):
        # Conf matrix
        self.plot_confusion_matrix(self.cm,'Captures/confusion_'+self.model_name+'_'+self.data_actual, self.class_names  )

        # Losses and accus
        self.plot_losses(self.results, 'Captures/losses_'+self.model_name+'_'+self.data_actual)

        # Save vectors of losses and accus
        self.save_results_disk('Results/res_'+self.model_name+'_'+self.data_actual)


    def save_results_disk(self, filename):
        np.save(filename, self.results)

    def plot_confusion_matrix(self, matrix  ,filename , labels , title='Confusion matrix'):
        fig, ax  = plt.subplots()
        ax.set_xticks([x for x in range(len(labels))])
        ax.set_yticks([y for y in range(len(labels))])
        # Place labels on minor ticks
        ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
        ax.set_xticklabels(labels, rotation='90', fontsize=10, minor=True)
        ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
        ax.set_yticklabels(labels[::-1], fontsize=10, minor=True)
        # Hide major tick labels
        ax.tick_params(which='major', labelbottom='off', labelleft='off')
        # Finally, hide minor tick marks
        ax.tick_params(which='minor', width=0)

        # Plot heat map
        proportions = [1. * row / sum(row) for row in matrix]
        ax.pcolor(np.array(proportions[::-1]), cmap=plt.cm.Blues)

        # Plot counts as text
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                confusion = matrix[::-1][row][col]
                if confusion != 0:
                    ax.text(col + 0.5, row + 0.5, confusion, fontsize=9,
                        horizontalalignment='center',
                        verticalalignment='center')

        # Add finishing touches
        ax.grid(True, linestyle=':')
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(filename+'.png')
        #cleaning memory
        plt.cla()
        plt.clf()
        plt.close(fig)
        return filename+'.png'

    def plot_losses(self, data , filename ):

        #plt.subplot(2,1,1)
        plt.figure()
        plt.plot(data['losses']['train'], c = 'b', label = 'train')
        plt.plot(data['losses']['val'], c = 'r', label = 'validation')
        plt.title('Loss 1: logMSE')
        plt.legend(loc='upper right')
        '''
        plt.subplot(2,1,2)
        plt.plot(data['acc']['train'], c = 'b', label = 'train')
        plt.plot(data['acc']['val'], c = 'r', label = 'validation')
        plt.legend(loc='upper right')
        plt.title('Accuracy')
        '''
        plt.savefig(filename+'.png')
        #cleaning memory
        plt.cla()
        plt.clf()
        plt.close()
        return filename+'.png'