# Sequence data generation for sequence-to-sequence sleep staging with XSleepNet
# X1: raw input
# X2: time-frequency input
# label: labels of sleep stage
# y: one-hot encoding of labels
import numpy as np
import h5py

class DataGenerator3:
    def __init__(self, list_of_files, file_sizes, data_shape_2=np.array([29, 128]), seq_len = 20):

        # Init params
        self.list_of_files = list_of_files
        self.file_sizes = file_sizes

        #self.data_shape_1 = data_shape_1
        self.data_shape_2 = data_shape_2
        #self.X1 = None
        self.X2 = None
        self.y = None
        self.label = None

        self.boundary_index = np.array([])

        self.seq_len = seq_len
        self.Ncat = 5 # five-class sleep staging

        self.pointer = 0
        self.reduce_pointer = 0
        self.data_index = None
        self.reduce_data_index = None
        self.data_size = np.sum(self.file_sizes)

        self.reduce_factor = 10

        # read data from mat files in the list stored in the file 'filelist'
        self.read_mat_filelist()

    # read data from mat files in the list stored in the file 'filelist'
    def read_mat_filelist(self):
        #self.X1 = np.ndarray([self.data_size, self.data_shape_1[0]])
        self.X2 = np.ndarray([self.data_size, self.data_shape_2[0], self.data_shape_2[1]])
        self.y = np.ndarray([self.data_size, self.Ncat])
        self.label = np.ndarray([self.data_size])
        count = 0
        for i in range(len(self.list_of_files)):
            #X1, X2, y, label = self.read_mat_file(self.list_of_files[i].strip())
            X2, y, label = self.read_mat_file(self.list_of_files[i].strip())
            #self.X1[count : count + len(X1)] = X1
            self.X2[count : count + len(X2)] = X2
            self.y[count : count + len(X2)] = y
            self.label[count : count + len(X2)] = label
            # boundary_index keeps list of end-of-recording indexes that cannot constitute a full sequence
            self.boundary_index = np.append(self.boundary_index, np.arange(count, count + self.seq_len - 1))
            count += len(X2)
            #print(count)

        #print("Boundary indices")
        #print(self.boundary_index)
        # this is all data indexes that can be used as starting point of sequences
        #self.data_index = np.arange(len(self.X1))
        self.data_index = np.arange(len(self.X2))
        #print(len(self.data_index))
        # exclude those starting indices in the boundary list
        mask = np.in1d(self.data_index,self.boundary_index, invert=True)
        self.data_index = self.data_index[mask]
        #print(len(self.data_index))
        #print(self.X1.shape, self.X2.shape, self.y.shape, self.label.shape)
        #print(self.X2.shape, self.y.shape, self.label.shape)

        self.reduce_data_index = self.data_index[0 : : self.reduce_factor]

        print('data index')
        print(len(self.data_index))
        print('reduce data index')
        print(len(self.reduce_data_index))

    def read_mat_file(self,filename):
        """
        Read matfile HD5F file and parsing
        """
        # Load data
        data = h5py.File(filename,'r')
        data.keys()
        # X1: raw data
        #X1 = np.array(data['X1']) # raw input
        #X1 = np.transpose(X1, (1, 0))  # rearrange dimension
        # X2: time-frequency data
        X2 = np.array(data['X2']) # time-frequency input
        X2 = np.transpose(X2, (2, 1, 0))  # rearrange dimension
        X2 = X2[:,:,1:] # excluding 0-th element
        y = np.array(data['y']) # one-hot encoding labels
        y = np.transpose(y, (1, 0))  # rearrange dimension
        label = np.array(data['label']) # labels
        label = np.transpose(label, (1, 0))  # rearrange dimension
        label = np.squeeze(label)

        #return X1, X2, y, label
        return X2, y, label

    def normalize(self, meanX2, stdX2):
        # data normalization for time-frequency input here
        X2 = self.X2
        X2 = np.reshape(X2,(self.data_size*self.data_shape_2[0], self.data_shape_2[1]))
        X2 = (X2 - meanX2) / stdX2
        self.X2 = np.reshape(X2, (self.data_size, self.data_shape_2[0], self.data_shape_2[1]))

        
    def shuffle_data(self):
        """
        Random shuffle the data points indexes
        """
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(self.data_index))
        self.data_index = self.data_index[idx]

        idx = np.random.permutation(len(self.reduce_data_index))
        self.reduce_data_index = self.reduce_data_index[idx]

                
    def reset_pointer(self):
        """
        reset pointer to begin of the list (used after one training epoch)
        and shuffle data again
        """
        self.pointer = 0
        #self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) samples and labels
        """
        data_index = self.data_index[self.pointer:self.pointer + batch_size]

        #update pointer
        self.pointer += batch_size

        # after stack eeg, eog, emg, data_shape now has one more dimension
        #batch_x1 = np.ndarray([batch_size, self.seq_len, self.data_shape_1[0], self.data_shape_1[1]])
        batch_x2 = np.ndarray([batch_size, self.seq_len, self.data_shape_2[0], self.data_shape_2[1], self.data_shape_2[2]])
        batch_y = np.ndarray([batch_size, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([batch_size, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                #batch_x1[i, n]  = self.X1[data_index[i] - (self.seq_len-1) + n, :, :]
                batch_x2[i, n]  = self.X2[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        #batch_x1.astype(np.float32)
        batch_x2.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        #return batch_x1, batch_x2, batch_y, batch_label
        return batch_x2, batch_y, batch_label

    # get rest of data that is collectively smaller than 1 full batch
    # this necessary for testing
    def rest_batch(self, batch_size):

        data_index = self.data_index[self.pointer : len(self.data_index)]
        # actual length of this batch
        actual_len = len(self.data_index) - self.pointer

        # update pointer
        self.pointer = len(self.data_index)

        # after stack eeg, eog, emg, data_shape now has one more dimension
        #batch_x1 = np.ndarray([actual_len, self.seq_len, self.data_shape_1[0], self.data_shape_1[1]])
        batch_x2 = np.ndarray([actual_len, self.seq_len, self.data_shape_2[0], self.data_shape_2[1], self.data_shape_2[2]])
        batch_y = np.ndarray([actual_len, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([actual_len, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                #batch_x1[i, n]  = self.X1[data_index[i] - (self.seq_len-1) + n, :, :]
                batch_x2[i, n]  = self.X2[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        #batch_x1.astype(np.float32)
        batch_x2.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        # return array of images and labels
        #return actual_len, batch_x1, batch_x2, batch_y, batch_label
        return actual_len, batch_x2, batch_y, batch_label


    def reset_reduce_pointer(self):
        self.reduce_pointer = 0

    def next_batch_reduce(self, batch_size):
        """
        This function gets the next n ( = batch_size) samples and labels
        """
        data_index = self.reduce_data_index[self.reduce_pointer:self.reduce_pointer + batch_size]

        #update pointer
        self.reduce_pointer += batch_size

        # after stack eeg, eog, emg, data_shape now has one more dimension
        #batch_x1 = np.ndarray([batch_size, self.seq_len, self.data_shape_1[0], self.data_shape_1[1]])
        batch_x2 = np.ndarray([batch_size, self.seq_len, self.data_shape_2[0], self.data_shape_2[1], self.data_shape_2[2]])
        batch_y = np.ndarray([batch_size, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([batch_size, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                #batch_x1[i, n]  = self.X1[data_index[i] - (self.seq_len-1) + n, :, :]
                batch_x2[i, n]  = self.X2[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        #batch_x1.astype(np.float32)
        batch_x2.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        #return batch_x1, batch_x2, batch_y, batch_label
        return batch_x2, batch_y, batch_label

    # get rest of data that is collectively smaller than 1 full batch
    # this necessary for testing
    def rest_batch_reduce(self, batch_size):

        data_index = self.reduce_data_index[self.reduce_pointer : len(self.reduce_data_index)]
        # actual length of this batch
        actual_len = len(self.reduce_data_index) - self.reduce_pointer

        # update pointer
        self.reduce_pointer = len(self.reduce_data_index)

        # after stack eeg, eog, emg, data_shape now has one more dimension
        #batch_x1 = np.ndarray([actual_len, self.seq_len, self.data_shape_1[0], self.data_shape_1[1]])
        batch_x2 = np.ndarray([actual_len, self.seq_len, self.data_shape_2[0], self.data_shape_2[1], self.data_shape_2[2]])
        batch_y = np.ndarray([actual_len, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([actual_len, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                #batch_x1[i, n]  = self.X1[data_index[i] - (self.seq_len-1) + n, :, :]
                batch_x2[i, n]  = self.X2[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        #batch_x1.astype(np.float32)
        batch_x2.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        # return array of images and labels
        #return actual_len, batch_x1, batch_x2, batch_y, batch_label
        return actual_len, batch_x2, batch_y, batch_label
