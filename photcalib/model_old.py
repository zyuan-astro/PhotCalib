import numpy as np


semester_list = np.array(["15A", "16A", "16B", "17A", "17B", "18A", "18B", "19A", "19B","20A", "20B","batch2", "batch3", "batch4", "batch5"])
x_center_list = np.array([9000, 10000, 10100, 10200, 9700, 9900, 9900, 9700, 10000, 10000, 9400, 9900, 9800, 10300, 9400])
y_center_list = np.array([9400, 10800, 11500, 10800, 11000, 11100, 11000, 11100, 10800, 10800, 10700, 10500, 10700, 10800, 10600])
amplitude_list = np.array([-0.23, 0.20, 0.13, 0.20, 0.24, 0.19, 0.19, 0.19, 0.18, 0.18, 0.20, 0.19, 0.19, 0.23, 0.11])
power_list = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
# center_offset_list = np.array([0.09, -0.03, -0.02, -0.03, -0.04, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.04, -0.02])


class FovOldModel():
    
    def __init__(self, semester):
        self.semester = semester
        
        ind = np.in1d(semester_list, self.semester)
        self.xc = x_center_list[ind]/19000.0
        self.yc = y_center_list[ind]/19000.0
        self.z_amp = amplitude_list[ind]
        self.power = power_list[ind]
#         best_center_offset = center_offset_list[ind]
        print (self.xc)
        
    def calib(self,x,y):
        
        dz = self.z_amp*np.power(np.sqrt((x-self.xc)**2.+(y-self.yc)**2.),self.power);
        
    
        return dz
        
