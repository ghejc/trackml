from __future__ import print_function
import numpy as np
import pandas as pd
from trackml.dataset import load_dataset
from trackml.score import score_event
from GaussianMixtureDensityNetwork import GaussianMixtureDensityNetwork
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class TrackFinder:
    
    MaxDetectorSize = 3000.0 # in mm
    
    class CircularBuffer:
        
        def __init__(self, size):
            assert size > 0
            self.buffer = list()
            self.size = size
            
        def __len__(self):
            "Number of elements in the buffer"
            return len(self.buffer)
        
        def add(self, elem):
            "Add an element to the buffer"
            if len(self.buffer) >= self.size:
                self.buffer.pop(0)
            self.buffer.append(elem)
            
        def values(self):
            "Returns the buffer values as 1-dim Numpy array"
            return np.array(self.buffer).flatten()
            
    
    def __init__(self, max_net = 5, logp_thresh = -10):
        """
        Creates a dictionary of networks calculating logarithm of p(hit belongs to track | previous hits in the track)
        Input dimension is (x,y,z,[volume,layer,module]) * n + q, where n is the number of previous hits
        Output dimension is (x,y,z,[volume,layer,module])
        """
        self.nets = dict()
        self.data = dict()
        self.input_columns = ['x','y','z','volume_id','layer_id']
        self.output_columns = ['x','y','z']
        self.log_prob = {-1 : None, 1 : None}
        self.logp_thresh = logp_thresh
        self.max_net = max_net
        
    def get_net(self,i):
        " Creates a new net if not present, otherwise returns the existing one"
        if i not in self.nets.keys():
            self.nets[i] = GaussianMixtureDensityNetwork(input_dim=len(self.input_columns)*i+1,output_dim=len(self.output_columns),num_distributions=100)        
        return self.nets[i]
    
    def get_data(self,i):
        " Creates a new data entry if not present, otherwise returns the existing one"        
        if i not in self.data.keys():
            self.data[i] = {'input':list(),'output':list()}
        return self.data[i]
    
    def make_train_data(self, tracks, hits, charges):
        " Adds the event data to the batch data"
        for track in tracks.keys():
            l0 = np.array([charges[track]])
            buf = TrackFinder.CircularBuffer(size=self.max_net)
            for hit in tracks[track]:
                i = len(buf)
                l = np.append(l0, buf.values(), axis=0)
                self.get_data(i)['input'].append(l)
                self.get_data(i)['output'].append(hits.loc[hit,self.output_columns].values)
                buf.add(hits.loc[hit,:].values)

    @staticmethod               
    def distance(x,y):
        "Distance between two hit coordinates"
        r = y[0:3] - x[0:3]
        return np.sqrt(np.sum(r*r))
        
    @staticmethod
    def random_choice(l, size):
        if isinstance(l, list):
            return np.random.choice(l, size, replace = False)
        else:
            return np.random.choice(list(l), size, replace = False)

    def validate_nets(self, tracks, hits, charges, size=10):
        """ 
        Applies nets to a random selection of tracks and saves 3D plots
        of predicted and actual track, where the predicted track has
        a color map proportional to the probability of the prediction.
        """
        for track in TrackFinder.random_choice(tracks.keys(), size):
            l0 = np.array([charges[track]])
            buf = TrackFinder.CircularBuffer(size=self.max_net)
            x = np.zeros((len(tracks[track]) + 1,3))
            logp = np.zeros((len(tracks[track]) + 1))
            n = 0
            for hit in tracks[track]:
                i = len(buf)
                if i not in self.nets.keys():
                    break
                l = np.append(l0, buf.values(), axis = 0)
                net = self.nets[i]
                y = hits.loc[hit,:].values
                buf.add(y)
                if i > 0:
                    x[n,:] = net.predict(l)
                else:
                    # the first hit is not predicted
                    x[n,:] = y[0:3]
                logp[n] = net.predict_log_proba(l, x[n,:])
                n = n + 1 
            i = len(buf)
            if i in self.nets.keys():
                l = np.append(l0, buf.values(), axis = 0)
                net = self.nets[i]
                x[n,:] = net.predict(l)
                logp[n] = net.predict_log_proba(l, x[n,:])
            # 3D plot with actual and predicted track
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x[:,0], x[:,1], x[:,2], c=logp, marker='x', cmap='jet')
            plot = ax.scatter(hits.loc[tracks[track],'x'].values, 
                       hits.loc[tracks[track],'y'].values,
                       hits.loc[tracks[track],'z'].values,
                       c='red', marker='+')
            ax.set_title('Track ' + str(track) + ' with charge ' + str(charges[track]))
            m = cm.ScalarMappable(cmap=plot.cmap)
            m.set_array(logp)
            cb = plt.colorbar(m)
            cb.set_label('Log Probability Density')
            plt.savefig('track_' + str(track) + '.png', dpi=400, bbox_inches='tight')
            # plt.show()
            
   
    def train(self, path='../data/train/', batch_size = 10000, validation_split=(99,100), number_of_events = np.inf):
        """
        Loads all training data, generates a dictionary of tracks with a ordered
        list of hits and feeds the data to a list of Keras models
        validation_split = (number of events used for training, number of events in a training-validation cycle)
        """
        n_events = 0
        train = True
        for event, hits, particles, truth in load_dataset(path,parts = ['hits','particles','truth']):
            hits = hits.set_index('hit_id')[self.input_columns]
            # normalize the hit coordinates
            hits.loc[:,'x':'z'] = hits.loc[:,'x':'z'] / TrackFinder.MaxDetectorSize
            truth = truth.set_index('hit_id')
            truth.loc[:,'tx':'tz'] = truth.loc[:,'tx':'tz'] / TrackFinder.MaxDetectorSize
            particles = particles.set_index('particle_id')
            particles.loc[:,'vx':'vz'] = particles.loc[:,'vx':'vz'] / TrackFinder.MaxDetectorSize
            tracks = dict()
            charges = dict()
            self.log_prob[1] = None
            self.log_prob[-1] = None
            
            train = (n_events % validation_split[1]) < validation_split[0]              
            
            # create a ordered list of hits for each track
            for particle in set(truth.particle_id[truth.particle_id > 0]):
                df = truth[truth.particle_id == particle].drop(columns = ['tpx','tpy','tpz','particle_id','weight'])
                x0 = particles.loc[particle,'vx':'vz'].values.reshape((1,-1))
                x = df.values - x0
                r2 = np.sum(x*x,axis = 1)
                # r2 is the squared distance to the original particle position
                df['r2'] = r2
                tracks[particle] = df.sort_values('r2', axis = 0).index.values
                charges[particle] = particles.loc[particle,'q']
 
            print ("Found " + str(len(tracks.keys())) + " tracks in event " + str(event))

            if train:
                # training phase
                self.make_train_data(tracks,hits,charges)
                # feed nets if enough data has accumulated
                for i in self.data.keys():
                    x = np.array(self.data[i]['input'])
                    y = np.array(self.data[i]['output'])
                    if y.shape[0] > batch_size:
                        print ("Fitting net " + str(i))
                        # check for NaN's in train data
                        #if np.any(np.isnan(x)):
                        #    print ("NaN in input")
                        #if np.any(np.isnan(y)):
                        #    print ("NaN in output")
                        self.get_net(i).partial_fit(x, y)
                        self.data[i]['input'] = list()
                        self.data[i]['output'] = list()
            else:
                # validation phase
                self.validate_nets(tracks,hits,charges)
                # calculate score
                # score = self.get_score(hits,truth)
                # print ("Score after " + str(n_events) + " is " + str(score))
                        
            n_events = n_events + 1
            if n_events >= number_of_events:
                break
            
    def get_score(self, hits, truth):
        tracks_pred = []
        while len(hits.index) > 0:
            self.find_track(hits, tracks_pred)
        submission = self.make_submission(tracks_pred)
        score = score_event(truth, submission)
        return score
            
    def test(self, path='../data/test/', number_of_events = np.inf):
        "Loads all test data (125 events)"
        n_events = 0
        tracks = dict()
        for event, hits in load_dataset(path, parts = ['hits']):
            hits = hits.set_index('hit_id')[self.input_columns]
            # normalize the hit coordinates
            hits.loc[:,'x':'z'] = hits.loc[:,'x':'z'] / TrackFinder.MaxDetectorSize
            tracks[event] = list()
            # apply tracking algorithm
            while len(hits.index) > 0:
                self.find_track(hits,tracks[event])
            print (str(len(tracks[event])) + " tracks found in event " + str(event))

            n_events = n_events + 1
            if n_events >= number_of_events:
                break
                    
        return tracks
    
    def find_track(self, hits, tracks_per_event):
        """
        Find all tracks for a event and adds them to the list tracks_per_event.
        The hits DataFrame will be modified within the function.
        All hits, which are assigned to a track, will be removed.
        """ 
        track = []
        logp = 0
        (hit, q) = self.find_first_hit(hits)
        track.append(hit)
        while True:
            (hit,logp) = self.find_next_hit(hits, track, q)
            # stop until logp drops under threshold
            if logp <= self.logp_thresh:
                break
            track.append(hit)
                
        if len(track) > 0:
            tracks_per_event.append(track)
            hits.drop(track, axis=1, inplace=True, errors='ignore')
            return True
        else:
            # hits.drop([hit], axis=1, inplace=True, errors='ignore')
            return False
            
    def get_nearest_hits(self, hits, track, x0 = None, r = 0.03):
        """
        Select hits closest to the track or x0 excluding hits,
        which are already in the track
        """
        if x0 is None:
            if len(track) == 0:
                return hits.index
            x0 = hits.loc[track[-1],'x':'z']
        else:
            x0 = pd.DataFrame(x0.reshape((1,-1)), columns=('x','y','z'))
        x = hits.loc[:,'x':'z'] - x0
        x2 = x.multiply(x).sum(axis=1)
        r2 = r * r
        return x[x2 < r2].index.drop(track, errors='ignore')
    
    def find_first_hit(self, hits):
        """"
        Returns the first hit of a possible track and caches
        the calculated log probabilities for the use in other
        tracks
        """
        net = 0
        logp_max = -np.inf
        next_hit = None
        q_hit = 1
        # search for the hit and q with the highest log probability
        if net in self.nets.keys():
            for q in (-1,1):
                x = np.array([q])
                if self.log_prob[q] is None:
                    y = hits.loc[:,self.output_columns].values
                    logp = self.nets[net].predict_log_proba(x,y)
                    self.log_prob[q] = pd.Series(logp, index = hits.index)
                else:
                    logp = self.log_prob[q][hits.index].values
                idx = np.argmax(logp)
                if logp[idx] > logp_max:
                    logp_max = logp[idx]
                    next_hit = hits.index[idx]
                    q_hit = q 
        return (next_hit, q_hit)
                             
    def find_next_hit(self, hits, track, q, r = 0.03):
        "Returns the most likely next hits together with the log probability"    
        logp_max = -np.inf
        next_hit = None
        net = len(track)
        if net > self.max_net:
            track = track[-self.max_net:]
            net = self.max_net
        if net in self.nets.keys():
            x = np.append(q, hits.loc[track,:].values.flatten())
            # the prediction is not a valid hit
            x_max = self.nets[net].predict(x)
            # find all hits within a certain spatial radius of the prediction
            nearest_hits = self.get_nearest_hits(hits,track,x_max,r)
            if not nearest_hits.empty:
# FutureWarning:
# Passing list-likes to .loc or [] with any missing label will raise
# KeyError in the future, you can use .reindex() as an alternative.
# 
# See the documentation here:
# https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex
# -listlike
                y = hits.loc[nearest_hits,self.output_columns].values
                # calculate the log probability of all hits near to the prediction
                logp = self.nets[net].predict_log_proba(x,y)
                # select the hit with the highest log probability
                idx = np.argmax(logp)
                logp_max = logp[idx]
                next_hit = hits.index[idx]
        return (next_hit, logp_max)
    
    def save(self, name):
        """
        Save all neural networks in a file
        """
        for net in self.nets.keys():
            self.nets[net].save(file_name = name + str(net))
 
    def make_submission(self, tracks):
        "Returns a DataFrame in the scoring submission format"
        data = []
        track_id = 0
        for track in tracks:
                for hit in track:
                    data.append([hit,track_id])
                track_id = track_id + 1
        return pd.DataFrame(data, columns = ('hit_id','track_id'))
            
    def make_submission_file(self, tracks, file_name='submission.csv'):
        "Saves the tracks in a file in file submission format"
        data = []
        for event in tracks.keys():
            track_id = 0
            for track in tracks[event]:
                for hit in track:
                    data.append([event,hit,track_id])
                track_id = track_id + 1
        submission = pd.DataFrame(data, columns = ('event_id','hit_id','track_id'))
        submission.to_csv(file_name, index = False)

if __name__ == '__main__':
    tf = TrackFinder()
    tf.train(validation_split=(2,3), number_of_events = 200)
    tf.save('tf')
    tf.make_submission_file(tf.test(number_of_events = 10))