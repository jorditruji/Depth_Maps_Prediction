
import os.path

class Helper():
    def __init__(self, session_name, img_folder, history_file):

        self.session_name = session_name
    	self.img_folder = img_folder
    	self.history_file = self.load_history(history_file)

    def load_history(self,file):
    	# Load experiments history
    	if os.path.isfile(file):
    		return read(file)
    	else:
    		return create_file(file)


    def write_experiment(self, model_name, config, results):
    	'''write new experiment data'''
    	img_paths = self.make_imgs(results)
    	time_now = datetime.now()
    	insert_struct = {}
    	data = {}
    	data['config'] = config
    	data['results'] = results
    	data['image_path'] = img_paths
    	insert_struct[model_name+str(time_now)]=data
    	self.history_file.append(insert_struct)


    def make_imgs(self,results):
    	'''plot images and store them to self.img_path'''
