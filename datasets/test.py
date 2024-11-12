import glob

src_path = './UCF-101'
vid_list = glob.glob(src_path + '/*/*.' + 'avi')

print (vid_list[0:15])
