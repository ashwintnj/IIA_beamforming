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


cp.get_default_memory_pool().free_all_blocks()   #command to free gpu memory
start_time_exe = time.time()  # start time of file

def select_files():  #function to select files
    # Create the root window for the file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select multiple .pcap files
    return filedialog.askopenfilenames(
        title="Select .pcap files", 
        filetypes=[("PCAP files", "*.pcap")]
    )
    
def calculate_time_axis(file_name,sun_file_diff):  #function to calculate time axis
    #------------TIME AXIS IN DECIMAL HRS CALCULATION------------------------------

    file_time=0.048  #2.88 min in hrs

 

    time_str = file_name.split("_")[1][8:14]

    t_hh=int(time_str[:2])
    t_mm=int(time_str[2:4])
    t_ss=int(time_str[4:6])
    
    file_start=t_hh+(t_mm/60)+(t_ss/3600)
    file_no= int(file_name[-len(str(sun_file_diff)):])-1-sun_file_diff  # change 4 or 3
    file_size=1900000
    start_time=file_start+(file_time*file_no)
    end_time=file_start+(file_time*(file_no+1))
    t_axis= np.linspace(start_time,end_time,file_size)
    
    
    return t_axis

def sun_dec(file_name):  #sun's declination find using ephem
    try:
        date_str = file_name.split("_")[1][:8] # Extract YYYYMMDD
        time_str = file_name.split("_")[1][8:14] # Extract HHMMSS

        dt_str = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        observer = ephem.Observer()
        observer.date = dt_str # Set observer's time

        sun = ephem.Sun(observer) # Compute Sun's position
        return float(sun.dec) * (180.0 / ephem.pi) # Convert to degrees
    except Exception as e:
        print(f"Error: {e}")
        return None

def time_delay(Decl):  #time delay calculations
    #print(f'Suns Declination: {Decl}')
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
    
    array_lat=13.6
    
    steering_angle = Decl - array_lat
    steering_angle_rad = np.deg2rad(steering_angle)
    print(f'Steer angle:{steering_angle}')
    # Delay
    
    #tow_instrument=np.array([0.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0])* 11.11e-9 # instrument delay ADC #0 1 clk delay
    tow_instrument=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])* 11.11e-9 #instrumental delay
    
    tow_g = d * np.sin(steering_angle_rad) / c  # (8,1) 
    towg=tow_instrument+tow_g   #instrumental delay and delay(formula)
    towg_reversed = towg[::-1]
    
    shi = (2 * np.pi * f_tot * towg[:,None]) # (num_frequencies, 8) 
    shi_1=(2 * np.pi * f_tot * towg_reversed[:,None])  #converting to 2*pi*f*t form
    complex_values=np.exp(1j*shi)
    complex_values_reversed=np.exp(1j*shi_1)  #converting to exponential form
    
    return steering_angle,steering_angle_rad,complex_values_reversed

def process_frames(file_path,complex_values_reversed_list,n_avg,n_avg_sub):
    frame_size=8330   #no. of bytes per frame including frame headers
    init_skip=start_index*frame_size*100
    freq_bins=4096
    batch_size=100  #reading 100 frames as a batch every time for GPU parallel processing
    
    n_frame=(stop_index-start_index)*100
    
    #creating empty arrays
    spec_2d = [np.zeros([freq_bins,n_frame//n_avg]) for _ in range(len(complex_values_reversed_list))]
    #spec_2d_db = [np.zeros([freq_bins,n_frame//n_avg]) for _ in range(len(complex_values_reversed_list))]
    #spec_2d_abs_sub = [np.zeros([freq_bins,n_frame//n_avg]) for _ in range(len(complex_values_reversed_list))]
    accumulated_spec=[cp.zeros([freq_bins,n_avg],dtype=cp.complex128) for _ in range(len(complex_values_reversed_list))]
    #avg_first_100_spectrums_abs=[np.zeros(freq_bins) for _ in range(len(complex_values_reversed_list))]
    
    f = open(file_path, 'rb')  #opening file as read binary
    f.seek(24)   #skipping file header
    
    f.seek(init_skip)  #skipping initial indices
    
    #---------------------------reference of LUT-----------------------------------
    ref_in=cp.arange(256)
    ref_data=cp.zeros((256,8),dtype=int)
    for n in range(256):
        ref_data[n, :] = cp.asarray([(ref_in[n] >> (7 - i)) & 1 for i in range(8)])
        
    ref_data=2*ref_data-1
    #------------------------------------------------------------------------------
    
    for frame in range(0,n_frame,batch_size):  
            batch_data = cp.zeros((batch_size, 8192), dtype=cp.uint8)    #creating dummy array for batch data
            for i in range(batch_size):  #read batch_size times in loop and create a 2d array as (100,8192)
                f.seek(138,1)   #skipping 138 file headers
                batch_data[i,:]=cp.asarray(cp.frombuffer(f.read(8192), dtype=cp.uint8)) #reading 8192 bytes and storing in batch_data

            ch_data=ref_data[batch_data,:8]  #using LUT reference converting into 1s and -1s

            # taking FFT for every channel seperately
            channel_1=cp.fft.fft(ch_data[:,:,7],axis=1) #S1
            channel_2=cp.fft.fft(ch_data[:,:,6],axis=1) 
            channel_3=cp.fft.fft(ch_data[:,:,5],axis=1)
            channel_4=cp.fft.fft(ch_data[:,:,4],axis=1)
            channel_5=cp.fft.fft(ch_data[:,:,3],axis=1)
            channel_6=cp.fft.fft(ch_data[:,:,2],axis=1)
            channel_7=cp.fft.fft(ch_data[:,:,1],axis=1)
            channel_8=cp.fft.fft(ch_data[:,:,0],axis=1) #N1 

            #converting the fft data into an array
            channel_data = cp.array([channel_1, channel_2, channel_3, channel_4, channel_5, channel_6, channel_7, channel_8])

            #taking 2nd half of fft bins
            channel_data_bins=channel_data[:,:,4096:]
            
            #multiple angle calculations
            for i,complex_values_reversed in enumerate(complex_values_reversed_list):
                # reshaping the time delay array by adding one new axis means converting from (8,4096) to (8,dummy,4096)
                complex_values_reversed_reshaped = cp.tile(complex_values_reversed[:, cp.newaxis, :], ( 1,100, 1))
    
                #multiplying channel data bins with time delay 
                channel_data_complex=channel_data_bins*cp.asarray(complex_values_reversed_reshaped)
    
                #summing the 8 channels after multipying with time delay
                accumulated_spec[i][:,:]= cp.sum(channel_data_complex, axis=0).T
                
                #averaging every 100 frames and making as 1 spectrum
                avg_spectra = cp.mean(cp.abs(accumulated_spec[i][:,:]), axis=1)
                
                
                spec_2d[i][:, frame//n_avg] = avg_spectra.get()
                #print(np.asarray(spec_2d).shape)
                # if spec_2d is mentioned as cp array then time will be some more less but gpu memory will occupy
                #if np array used for spec_2d memory will occupy in cpu but 1 min time will increase
            print((frame//n_avg)+1)
            

            
            
            '''frame_counter+=1
            #averging n_avg 100 after absolute
            print(frame_counter)
            if (frame + 1) % n_avg== 0:
                avg_spectra = cp.mean(cp.abs(accumulated_spec), axis=1)
                #fft_magn = 10 * np.log10(avg_spectra) +1e-10 #1e-10 to handle divide by zero warnings
                fft_magn_scaled = cp.array(avg_spectra)
                
                #Save the averaged spectrum
                spec_2d[:, frame//n_avg] = fft_magn_scaled
                
                print(spec_2d)
                #Reset the accumulator for the next set of frames
                #accumulated_spec= np.zeros([4096,n_skip],dtype=np.int8)
                print((frame//n_avg)+1)
                print('-----------------------------------------------------------')'''
    print(f'{steering_angle}_completed')
    
    #writing spec_2d into binary file
    for i,output_name in enumerate(output_name_list):
        #-------------------WRITING TO BIN FILE--------------------------------
        # writing spec_2d needs to take bg sub and 10log10 after reading bin file
        array_size=np.size(spec_2d[i])
        os.makedirs(output_folder_bin, exist_ok=True)
        binary_output_file_path = os.path.join(output_folder_bin, f'{output_name}_{array_size}_array_size_float32.bin')
        spec_2d[i][fmin:fmax,:].astype(np.float32).tofile(binary_output_file_path)
        #----------------------------------------------------------------------
    end_time_exe=time.time()
    execution_time = end_time_exe - start_time_exe
    print(f"Execution time: {execution_time} seconds")
    return spec_2d
   

if __name__ == "__main__":
    start_time_exe=time.time()
    
    file_paths=select_files()  #fn to select files
    
    for file_path in file_paths:
        
        cp.get_default_memory_pool().free_all_blocks()   #command to free gpu memory
        file_name = os.path.basename(file_path).split('.')[0] + '_' + os.path.basename(file_path).split('.')[1]
        
        Decl=sun_dec(file_name)  #fn to find sun's Declination

        no_of_angles=10  #give total number needed/2 if 20 needed then give 10
        
        # Create a list for the angles
        angles = [Decl]  # Start by adding the initial angle
        
        step_size=4  #step size of angle
        
        # Multiple angles 
        # Add n positive steps
        for i in range(no_of_angles):
            angles.append(Decl + (i + 1) * step_size)

        # Add n negative steps
        for i in range(no_of_angles):
            angles.append(Decl - (i + 1) * step_size)
        
        angles=[-50.4]
        #angles=np.append(np.linspace(-23,23,5),Decl)  #if u need from -23 to 23 with count of n angles use this
        #if multiple angles needed change above
        angles=np.sort(angles)  #sorting the angles in order
        
        complex_values_reversed_list = []
        actual_decl=Decl
        actual_steering_angle=Decl-13.6 
        
        for Decl in angles:
            cp.get_default_memory_pool().free_all_blocks()  #command to free gpu memory
            
            start_time_ang = time.time()
            print(f'Suns Declination: {Decl}')
            steering_angle,steering_angle_rad,complex_values_reversed=time_delay(Decl)
            
            complex_values_reversed_list.append(complex_values_reversed)
            
        # enter sun file diff means if first file sun number is 2 means then 2-1=1 then enter sun file diff as 1
        #if sun number in file and starting file has same number means enter 0
        # enter diff of sun start value and normal count value on sun_file_diff
        #enter sun_file_diff as starting file-1
        #sun_file_diff=267  # on 18.3.25
        #sun_file_diff=833  # on 23.3.25
        #sun_file_diff=1487  # on 27.3.25
        #sun_file_diff=2003  # on 30.3.25
        #sun_file_diff=2168  # on 31.3.25
        sun_file_diff=2452  # on 2.4.25
        #sun_file_diff=0
        t_axis=calculate_time_axis(file_name, sun_file_diff)
        
        
        #for full file
        t_start=t_axis.min()
        t_end=t_axis.max()
        
        diff_start=abs(t_axis-t_start)
        index_start=diff_start.argmin()
        
        diff_end=abs(t_axis-t_end)
        index_end=diff_end.argmin()
                
        n_avg=100
        n_avg_sub=100
        
        
        #------finding start and stop using start and end time index-----------
        start_index=index_start//100
        stop_index=index_end//100
        no_frames=stop_index-start_index
        print(f'{no_frames} Loops going to run')
        #----------------------------------------------------------------------
        
        output_folder_bin = '/home/pulsar/Documents/intern_work/Observation_init_skip_images_time_axis_GPU/2025_04_02_multi_angles/bin_file'
        
        os.makedirs(output_folder_bin, exist_ok=True)
        
       
        fmin=0
        fmax=4096
        freq_bins=4096
        

        
        title_list=[]
        output_file_path_list=[]
        output_name_list=[]
        array_lat=13.6
        for i, Decl in enumerate(angles):
            
            steering_angle = Decl - array_lat
            title_name = (file_name + f'_start_time_{index_start}_end_time_{index_end}_{n_avg}_points_avg_(bg_sub_{n_avg_sub}_pts)\n'
                          f'(sky_position_{round(Decl,2)}°_steer_angle_{round(steering_angle,2)}°)_({no_frames}_pts_plot)_({start_index}_initial_skip)_(actual_delay)_(mean_after_abs)_10log10_plot_vmax_2\n'
                          f'plot_actual_Decl_{round(actual_decl,2)}°_GPU')
            
            title_list.append(title_name)
            
            output_name=f'{i}'+file_name + f'_sky_position_{round(Decl,2)}°_steer_angle_{round(steering_angle,2)}°_{no_frames}_loops_'
            output_name_list.append(output_name)
            

            
        freq_axis=np.linspace(45,90,freq_bins)
        time_axis=np.linspace(t_start,t_end,no_frames)
        freq_bin_axis=np.linspace(fmin,fmax,freq_bins)
        time_bin_axis=np.linspace(start_index,stop_index,no_frames)
        
        #all variables sending to process frames are in global scope. So no need to send all. Directly it will take
        spec_2d=process_frames(file_path, complex_values_reversed_list,n_avg,n_avg_sub)

            
    end_time_exe=time.time()
    
            
    #cp.get_default_memory_pool().free_all_blocks()  #command to free gpu memory
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
    execution_time = end_time_exe - start_time_exe
    print(f"Execution time: {execution_time} seconds")