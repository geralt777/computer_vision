import sys
import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle

def help_message():

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/aligned000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/aligned001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    
    color_img = cv2.resize(cv2.imread("images/aligned001.jpg", cv2.IMREAD_COLOR), (0,0), fx=scale_factor,fy=scale_factor)
    #get rgb values
    conn = []
    h1,w1,num = color_img.shape
    
    ref_avg   = (ref_white + ref_black) / 2.0
    
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region
    
    

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        
        patt_gray = cv2.resize(cv2.imread("images/aligned%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        #on_mask = on_mask.asType(int)
        

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        #on_mask = on_mask* bit_code
        
        #scan_bits = scan_bits + bit_code
        
        for x in range(h):
            for j in range(w):
                if on_mask[x][j] == True:
                    scan_bits[x][j] = scan_bits[x][j] | bit_code

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","rb") as f:
        binary_codes_ids_codebook = pickle.load(f, encoding='latin1')

    camera_points = []
    projector_points = []
    half_img = np.zeros((h,w,3), dtype=np.float)
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
            
            p_x, p_y = binary_codes_ids_codebook[scan_bits[y,x]] 
            Red = p_x
            Green = p_y
            Blue = 0
            rgb_sum = Red + Green + Blue;
            NormalizedRed = Red/rgb_sum;
            NormalizedGreen = Green/rgb_sum;
            NormalizedBlue = Blue/rgb_sum;
            if p_x >= 1279 or p_y >= 799: # filter
                continue
            half_img[y][x] = [0,p_y*255/799.0,p_x*255/1279.0]
            
            projector_points.append([p_x,p_y])
            camera_points.append([x/2,y/2])
            tempv = color_img[y][x]
            conn.append([tempv[2], tempv[1], tempv[0]])
            
            
    
    camera_mod = np.reshape(np.array(camera_points,dtype = np.float32),(len(camera_points),1,2))
    proj_mod = np.reshape(np.array(projector_points,dtype = np.float32),(len(projector_points),1,2))
        
    cv2.imwrite("correspondance.jpg", half_img)    
        

    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    
    idmat = np.eye(3,4)

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","rb") as f:
        d = pickle.load(f, encoding='latin1')
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']
        projector_RT = np.column_stack((projector_R,projector_t))
        
        qq = cv2.undistortPoints(camera_mod, camera_K, camera_d)
        pp=cv2.undistortPoints(proj_mod, projector_K, projector_d)
        t_p = cv2.triangulatePoints(idmat,projector_RT,qq,pp);
        points_3d = cv2.convertPointsFromHomogeneous(t_p.transpose())
        
    
    points_3d_color = np.append(points_3d, np.reshape(np.array(conn),(len(conn),1,3)),axis=2)
    mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
    indices = []
    for i in range(len(mask)):
        if mask[i] == False:
            indices.append(i)
    
    points_3d = np.delete(points_3d,indices,0)
    points_3d_color = np.delete(points_3d_color,indices,0)
    
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
        for p in points_3d_color:
            f.write("%d %d %d %d %d %d\n"%(p[0,0],p[0,1],p[0,2],p[0,3],p[0,4],p[0,5]))
    
    return points_3d
	
def write_3d_points(points_3d):
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))

    #return points_3d, camera_points, projector_points
    return
    
if __name__ == '__main__':
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()
    
    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
	
