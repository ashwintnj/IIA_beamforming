import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import time
import sys
import ephem
from matplotlib.gridspec import GridSpec
import cupy as cp

start_time = time.time()
# Create the root window for the file dialog
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select multiple .pcap files
file_paths = filedialog.askopenfilenames(
    title="Select .pcap files", 
    filetypes=[("PCAP files", "*.pcap")]
)

#--------------------------TIME DELAY CORRECTION-------------------------------
n = 8
d = np.array([0, 5, 10, 15, 20, 25, 30, 35])  # 8 antenna elements
c = 3e8
f1 = 45e6
f2 = 90e6

f_band = f2 - f1
freq_bins = 4096
n_add_freq = f_band / freq_bins  # 10.98 kHz frequency step
f_tot = np.arange(f1, f2, n_add_freq)

# Steering angle setup
#Decl=input('Enter the sun declination: ') #-3.678 on 11.03.2025
#Decl=-3.678 # on 11.03.2025
#Decl=-3.285 # on 12.03.2025
#Decl=-2.79  #on 13.03.2025
#Decl=-2.295 #on 14.03.2025 verify
#Decl=-2.04 #on 15.03.2025 verify
#----------------Declination find with filename--------------------------------
def sun_declination_pyephem(filename):
    """Computes the Sun's declination using PyEphem."""
    try:
        date_str = filename.split("_")[1][:8] # Extract YYYYMMDD
        time_str = filename.split("_")[1][8:14] # Extract HHMMSS

        dt_str = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        observer = ephem.Observer()
        observer.date = dt_str # Set observer's time

        sun = ephem.Sun(observer) # Compute Sun's position
        return float(sun.dec) * (180.0 / ephem.pi) # Convert to degrees
    except Exception as e:
        print(f"Error: {e}")
        return None
for file_path in file_paths: #,reverse=True 
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path).split('.')[0]+'_'+os.path.basename(file_path).split('.')[1]
    Decl = round(sun_declination_pyephem(file_name),2)
    
#------------------------------------------------------------------------------
array_lat=13.6
    
steering_angle = float(Decl) - array_lat

steering_angle_rad = np.deg2rad(steering_angle)

# Delay
towg = np.asarray(d[:, None]) * np.sin(steering_angle_rad) / c  # (8,1) 
towg_reversed = towg[::-1]

shi = (2 * np.pi * f_tot * towg) # (num_frequencies, 8) 
shi_1=(2 * np.pi * f_tot * towg_reversed)
complex_values=np.exp(1j*shi)
complex_values_reversed=np.exp(1j*shi_1)
#complex_values_t=complex_values.T
#complex_values_reversed_t=complex_values_reversed.T
#------------------------------------------------------------------------------
#---------------------------reference of LUT-----------------------------------
ref_in=cp.arange(256)
ref_data=cp.zeros((256,8),dtype=int)
for n in range(256):
    ref_data[n, :] = cp.asarray([(ref_in[n] >> (7 - i)) & 1 for i in range(8)])
   
ref_data=2*ref_data-1
#------------------------------------------------------------------------------

output_folder = '/home/pulsar/Documents/intern_work/Observation_quick_look_images_GPU/2025_04_25/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)  
for file_path in file_paths: #,reverse=True 
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path).split('.')[0]+'_'+os.path.basename(file_path).split('.')[1]
    
    frame_size = 8330  #  (58 + 80 + 8192)
    total_frames = (file_size - 24) // (frame_size)  
    
    f = open(file_path, 'rb')
    f.seek(24)  

    n_skip=1000

    fft_magn=[]
    fft_complex=[]
    n_frame=total_frames//n_skip
    n_frame=n_frame-10
    fmin=0
    fmax=4096   #mention -1 for 4096
    bg_sub=100
    #frame_indices=np.linspace(0,total_frames,n_frame,dtype=int)
    spec_2d = cp.zeros([freq_bins,n_frame])
    #f.seek(init_skip*frame_size) 
    for frame in range(n_frame):
        f.seek(n_skip*frame_size,1)
        f.seek(138,1)  #skipping current UDP header
        #content = f.read(8192)   #reading data
        content = cp.frombuffer(f.read(8192), dtype=cp.uint8) # reading data
        ch_data=ref_data[content,:8]
        channel_1=cp.fft.fft(ch_data[:,7])
        channel_2=cp.fft.fft(ch_data[:,6])
        channel_3=cp.fft.fft(ch_data[:,5])
        channel_4=cp.fft.fft(ch_data[:,4])
        channel_5=cp.fft.fft(ch_data[:,3])
        channel_6=cp.fft.fft(ch_data[:,2])
        channel_7=cp.fft.fft(ch_data[:,1])
        channel_8=cp.fft.fft(ch_data[:,0])
        channel_data = cp.array([channel_1, channel_2, channel_3, channel_4, channel_5, channel_6, channel_7, channel_8])
        
        #fft_result = np.fft.fft(channel_1_data)
        #channel_data_bins=channel_data.T[:,4096:]
        channel_data_bins=channel_data[:,freq_bins:]
        #channel_data_complex=channel_data_bins*complex_values
        channel_data_complex=channel_data_bins*cp.asarray(complex_values_reversed)

        fft_complex_sum=cp.sum(channel_data_complex,axis=0)
        fft_magn=10*cp.log10(cp.abs(fft_complex_sum))
        fft_magn_scaled=cp.array(fft_magn,dtype=np.int8)
        spec_2d[:,frame]=fft_magn_scaled[fmin:fmax]
        print(frame)
             
        
    

    
    f.close()
    
    os.makedirs(output_folder, exist_ok=True)
    output_name = file_name + f'\n{bg_sub}_background subracted_{n_skip}_frames_skipped_QUICK_LOOK_bgsub'
    output_file_path = os.path.join(output_folder, f'{output_name}.jpg')
    
    #------------first bg_sub row avg sub--------------------------------------
    bg_sub_spec_data = np.mean(spec_2d[:, :bg_sub], axis=1)
    spec_2d_abs_sub = spec_2d - bg_sub_spec_data[:, cp.newaxis]
    #--------------------------------------------------------------------------
    
    #-----------3072-4096 freq all row bg sub----------------------------------
    #bg_sub_freq=np.mean(spec_2d[3072:, :], axis=0)
    #spec_2d_abs_sub=spec_2d - bg_sub_freq[ np.newaxis,:]
    #--------------------------------------------------------------------------
    
    #spec_2d_abs_sub=spec_2d
    # 1d spect mult
    #col_summed_spectrum = np.sum(spec_2d, axis=0)  # Sum over columns
    col_summed_spectrum =np.nanmean(np.nan_to_num(spec_2d_abs_sub.get(), nan=np.nan, posinf=np.nan, neginf=np.nan),axis=0)
    row_summed_spectrum = np.mean(spec_2d_abs_sub.get(), axis=1)   # Sum over rows
    
    freq_axis=np.linspace(45,90,freq_bins)
    #time_axis=np.linspace(0,np.uint(n_frame/100),n_frame/100) #no of frames (n_frame-10)
    time_axis=np.linspace(0,n_frame,n_frame)
    freq_bin_axis=np.linspace(fmin,fmax,n_frame)

    # grid layout
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 3, 1], width_ratios=[3, 0.0, 1])
    
    
    ax1 = fig.add_subplot(gs[1, 2])
    ax1.plot(row_summed_spectrum[fmin:fmax], freq_axis[fmin:fmax])
    # ax1.set_xlabel('Power')
    # ax1.set_title('Summed by Row')
    #ax1.set_xlim(-0.5,5)
    ax1.set_ylim(freq_axis.min(),freq_axis.max())
    ax1.yaxis.tick_right()  # marking at right side
    ax1.yaxis.set_label_position("right")  
    ax1.set_ylabel('Frequency (MHz)')  
    
    
    ax2 = fig.add_subplot(gs[1, 0])
    im=plt.imshow(spec_2d_abs_sub[fmin:fmax,:].get(), aspect='auto', extent=[time_axis.min(), time_axis.max(), freq_axis[fmin], freq_axis[fmax-1]], origin='lower', cmap='jet')#cmap='Spectral',vmin=0.0,vmax=0.05
    #im.xaxis.tick_top()
    #im.xaxis.set_label_position("top")
    ax2.set_xlabel('Total No. of Frames')
    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_title(f'{output_name}')
    
    ax2_1 = ax2.twinx()  
    ax2_1.set_ylim(freq_bin_axis.min(),freq_bin_axis.max())
    ax2_1.set_ylabel('Frequency Bins')
    
    # 1D Spectrum (Summed over Columns) - Bottom
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(col_summed_spectrum)
    ax3.set_xlim(time_axis.min(),time_axis.max())
    #ax3.set_ylim(0,1)
    ax3.set_xlabel('Total No. of Frames')
    # ax3.set_ylabel('Power')
    # ax3.set_title('Summed by Column')
    
    # adjusting grid shape
    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    plt.savefig(output_file_path,bbox_inches='tight')
    
    plt.show() 
    plt.close()
end_time = time.time()

#-------------------------Variable Memory--------------------------------------
# variable size in MB
var= {name: sys.getsizeof(value) / (1024 * 1024)  for name, value in globals().items() if not name.startswith("__")} # converting into mb
    
# ascending order to descending order
#sorted_vars = sorted(var.items(), key=lambda x: x[1]) 
    
# descending to ascending order
sorted_vars = sorted(var.items(), key=lambda x: x[1], reverse=True)
    
#priniting variable in size 
print("\nvariable size(MB)-")
for name, size in sorted_vars:
    print(f"{name}: {size:.2f} MB")  # we can change flotting value
#------------------------------------------------------------------------------
        
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")