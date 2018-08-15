import numpy as np
import pandas as pd
from trackml.dataset import load_dataset
from trackml.score import score_event
from GaussianMixtureDensityNetwork import GaussianMixtureDensityNetwork

class TrackFinder:
    
    MaxDetectorSize = 3000.0 # in mm
    
    def __init__(self, logp_thresh = -10):
        """
        Creates a list of networks calculating logarithm of p(hit belongs to track | previous hits in the track)
        Input dimension is (x,y,z,[volume,layer,module]) * n + q, where n is the number of previous hits
        Output dimension is (x,y,z,[volume,layer,module])
        """
        self.nets = dict()
        self.data = dict()
        self.input_columns = ['x','y','z','volume_id','layer_id']
        self.output_columns = ['x','y','z']
        self.log_prob = {-1 : None, 1 : None}
        self.logp_thresh = logp_thresh
        
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
            l = np.array([charges[track]])
            i = 0
            for hit in tracks[track]:
                self.get_data(i)['input'].append(l)
                self.get_data(i)['output'].append(hits.loc[hit,self.output_columns].values)
                l = np.append(l, hits.loc[hit,:].values, axis=0)
                i = i + 1

    @staticmethod               
    def distance(x,y):
        "Distance between two hit coordinates"
        r = y[0:3] - x[0:3]
        return np.sqrt(np.sum(r*r))

    def validate_nets(self, tracks, hits, charges, size=10):
        """ 
        Applies nets to a random choice of tracks and returns a dictionary
        of tracks containing a list of tuples with the spatial distance between
        predicted and actual hit and the log probability of the
        actual hit including the stopping criterion (the next hit,
        where the log probability drops below the threshold).
        """
        result = dict()
        for track in np.random.choice(tracks.keys(), size):
            l = np.array([charges[track]])
            i = 0
            result[track] = list()
            for hit in tracks[track]:
                if i not in self.nets.keys():
                    break
                net = self.nets[i]
                if i > 0:
                    x = net.predict(l)
                else:
                    x = np.zeros((3))
                y = hits.loc[hit,:].values
                logp = net.predict_log_proba(l,hits.loc[hit,self.output_columns].values)
                r = TrackFinder.distance(y,x)
                result[track].append((logp,r))    
                l = np.append(l, y, axis = 0)
                i = i + 1
            if i in self.nets.keys():
                net = self.nets[i]
                x = net.predict(l)
                r = 0
                logp = net.predict_log_proba(l,x)
                result[track].append((logp,r))
        print result
    
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
 
            print "Found " + str(len(tracks.keys())) + " tracks in event " + str(event)

            if train:
                # training phase
                self.make_train_data(tracks,hits,charges)
                # feed nets if enough data has accumulated
                for i in self.data.keys():
                    x = np.array(self.data[i]['input'])
                    y = np.array(self.data[i]['output'])
                    if y.shape[0] > batch_size:
                        print "Fitting net " + str(i)
                        # check for NaN's in train data
                        #if np.any(np.isnan(x)):
                        #    print "NaN in input"
                        #if np.any(np.isnan(y)):
                        #    print "NaN in output"
                        self.get_net(i).partial_fit(x, y)
                        self.data[i]['input'] = list()
                        self.data[i]['output'] = list()
            else:
                # validation phase
                self.validate_nets(tracks,hits,charges)
                # calculate score
                tracks_pred = []
                while len(hits.index) > 0:
                    self.find_track(hits, tracks_pred)
                submission = self.make_submission(tracks_pred)
                score = score_event(truth, submission)
                print "Score after " + str(n_events) + " is " + str(score)
                print "Number of predicted tracks is " + str(len(tracks_pred))
                        
            n_events = n_events + 1
            if n_events >= number_of_events:
                break
            
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
            print str(len(tracks[event])) + " tracks found in event " + str(event)

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
        if net in self.nets.keys():
            x = np.append(q, hits.loc[track,:].values.flatten())
            # the prediction is not a valid hit
            x_max = self.nets[net].predict(x)
            # find all hits within a certain spatial radius of the prediction
            nearest_hits = self.get_nearest_hits(hits,track,x_max,r)
            if not nearest_hits.empty:
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
    tf.train(number_of_events = 200)
    tf.save('tf')
    tf.make_submission_file(tf.test(number_of_events = 10))