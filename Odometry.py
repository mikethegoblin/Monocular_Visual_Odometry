from glob import glob
import cv2, skimage, os
import numpy as np
from tqdm import tqdm

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))

        self.K, self.P = self.form_calib()
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
        
    def form_calib(self):
        '''
        This function forms the intrinsic parameter matrix and camera projection matrix
        '''
        K = np.array([
            [self.focal_length, 0, self.pp[0]],
            [0, self.focal_length, self.pp[1]],
            [0, 0, 1]
        ]).astype(np.float32)

        P = np.zeros((3, 4))
        P[:3, :3] = K
        return K, P

    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def load_img(self, frame_id):
        '''
        This function loads a frame as a gray scale image

        Parameters
        ----------
        frame_id (int): the index of a frame

        Returns
        -------
        image: a gray scale image
        '''
        return self.imread(self.frames[frame_id])

    def form_transformation(self, R, t):
        '''
        This function constructs the transformation matrix from R and t

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        '''
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return np.float32(T)


    def extract_features(self, img, detector='sift', mask=None):
        '''
        This function detects and computes potential keypoints in an image

        Parameters
        ----------
        detector (string): The type of the detector to be used
        mask: image mask

        Returns
        -------
        kp (ndarray): keypoints detected
        des (ndarray): keypoint descriptors
        '''
        if detector == 'sift':
            det = cv2.SIFT_create()
        elif detector == 'orb':
            det = cv2.ORB_create(nfeatures=3000)

        kp, des = det.detectAndCompute(img, mask)
        return kp, des

    def match_features(self, des1, des2, matching='BF', detector='sift', k=2):
        '''
        This function finds matches between two keypoint descriptors

        Parameters
        ----------
        des1 (ndarray): keypoint descriptor for img1
        des2 (ndarray): keypoint descriptor for img2
        matching (string): matching algorithm to be used
        detector (string): types of detector used in computing potential keypoints
        k (int): number of neighbors if use FLANN based matching

        Returns
        -------
        good_matches (list): list of good matches between two descriptors
        '''
        if matching == 'BF':
            if detector == 'sift':
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            elif detector == 'orb':
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif matching == 'FLANN':
            if detector == 'sift':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
            elif detector == 'orb':
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6, # 12
                        key_size = 12,     # 20
                        multi_probe_level = 1) #2
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        matches = matcher.knnMatch(des1, des2, k=k)

        # filter matches
        good_matches = []
        for pair in matches:
            if len(pair) >= 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        return good_matches

    def decomp_essential_mat(self, E, q1, q2):
        '''
        This function decomposes the essential matrix into R and t

        Parameters
        ----------
        E (ndarray): The essential matrix
        q1 (ndarray): good keypoint matches positions in i-1'th image
        q2 (ndarray): good keypoint matches positions in ith image

        Returns
        -------
        correct_pair (tuple): A tuple in the form of (R, t)
        '''
        def get_num_pos_zcord(R, t):
            # Get the transformation matrix
            T = self.form_transformation(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from second frame
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points that have positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        z_sums = []
        for R, t in pairs:
            z_sum = get_num_pos_zcord(R, t)
            z_sums.append(z_sum)

        # select the pair with the most points with positive z coordinates
        correct_pair_idx = np.argmax(z_sums)
        correct_pair = pairs[correct_pair_idx]
        R, t = correct_pair

        return R, t

    def get_gt(self, frame_id):
        '''
        This function gets the ground truth pose corresponding to a specific frame
        '''
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def get_matches(self, frame_id):
        '''
        This function matches keypoints between two consecutive frames

        Parameters
        ----------
        frame_id (int): index of current frame

        Returns
        -------
        q1 (ndarray): good keypoint matches positions in i-1'th image
        q2 (ndarray): good keypoint matches positions in ith image
        '''
        # load the two consecutive frames
        prev_frame = self.load_img(frame_id-1)
        cur_frame = self.load_img(frame_id)
        # find keypoints and descriptors
        kp1, des1 = self.extract_features(prev_frame, 'orb', None)
        kp2, des2 = self.extract_features(cur_frame, 'orb', None)

        # get good keypoint matches between two frames
        matches = self.match_features(des1, des2, 'FLANN', 'orb', 2)

        q1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        q2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return q1, q2

    def get_pose(self, q1, q2, frame_id):
        '''
        This function estimates the camera pose from matching key points between consecutive frames

        Parameters
        ----------
        q1 (ndarray): good keypoint matches positions in i-1'th image
        q2 (ndarray): good keypoint matches positions in ith image
        frame_id (int): index of current frame

        Returns
        -------
        transf_mat (ndarray): the transformation matrix that describes the cameara pose
        '''
        E, _ = cv2.findEssentialMat(q1, q2, focal=self.focal_length, pp=self.pp, method=cv2.RANSAC, prob=0.9995, threshold=0.59)

        # _, R, t, _ = cv2.recoverPose(E, q1, q2, focal=self.focal_length, pp=self.pp)
        # t = t * self.get_scale(frame_id)
        R, t = self.decomp_essential_mat(E, q1, q2)
        t = t * self.get_scale(frame_id)
        transf_mat = self.form_transformation(R, np.squeeze(t))
        return transf_mat

    
    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        estimated_path = []
        for i in tqdm(range(len(self.frames))):
            if i == 0:
                cur_pose = np.identity(4)
                # use first ground truth pose as initial pose
                cur_pose[:3, :] = np.array(self.pose[0]).reshape((3,4))
                translation = cur_pose[:3, 3]
            else:
                q1, q2 = self.get_matches(i)
                transf = self.get_pose(q1, q2, i)
                cur_pose = cur_pose @ np.linalg.inv(transf)
                translation = cur_pose[:3, 3]
            estimated_path.append([translation[0], translation[1], translation[2]])

        estimated_path = np.array(estimated_path)
        np.save('predictions.npy', estimated_path)

        return estimated_path
        

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)
