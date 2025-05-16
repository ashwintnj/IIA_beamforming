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

    time_str = file_name.split("_")[1][8:14]  #splitting the start UT time from filename

    t_hh=int(time_str[:2])   #splitting HH from time
    t_mm=int(time_str[2:4])  #splitting MM from time
    t_ss=int(time_str[4:6])  #splitting SS from time
    
    file_start=t_hh+(t_mm/60)+(t_ss/3600)    #converting to decimal hrs (HH:MM:SS to HH.HH)
    file_no=int(file_name.split("_")[5].replace('SUN',''))-1-sun_file_diff   #finding the file number to calculate time axis
    #file_no= int(file_name[-len(str(sun_file_diff)):])-1-sun_file_diff  # change 4 or 3  
    file_size=1900000   
    start_time=file_start+(file_time*file_no)  #calculating start time
    end_time=file_start+(file_time*(file_no+1)) #calculating end time
    t_axis= np.linspace(start_time,end_time,file_size)   #time axis
    
    
    return t_axis

def sun_dec(file_name):  #finding sun's declination using ephem
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

def time_delay(Decl):   #calculating time delay
    #print(f'Suns Declination: {Decl}')
    n = 8  # number of antennas
    d = np.array([0, 5, 10, 15, 20, 25, 30, 35])  # 8 antenna elements
    c = 3e8  #speed of light
    f1 = 45e6  #start frequency
    f2 = 90e6  #end frequency
     
    f_band = f2 - f1   #bandwidth
    freq_bins = 4096   #freq bins
    n_add_freq = f_band / freq_bins  # 10.98 kHz frequency step 
    f_tot = np.arange(f1, f2, n_add_freq)
    
    # Steering angle setup
    array_lat=13.6  #latitude of gauribidanur
    
    steering_angle = Decl - array_lat  #steering angle
    steering_angle_rad = np.deg2rad(steering_angle)
    print(f'Steer angle:{steering_angle}')
    # Delay
    
    #tow_instrument=np.array([0.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0])* 11.11e-9 # instrument delay ADC #0 1 clk delay
    tow_instrument=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])* 11.11e-9 #instrument delay
    
    tow_g = d * np.sin(steering_angle_rad) / c  # (8,1) 
    towg=tow_instrument+tow_g  #adding instrument delay with time delay
    towg_reversed = towg[::-1]  #reversing time delay 
    
    shi = (2 * np.pi * f_tot * towg[:,None]) # (num_frequencies, 8) 
    shi_1=(2 * np.pi * f_tot * towg_reversed[:,None])   #converting to 2*pi*f*t form
    complex_values=np.exp(1j*shi)
    complex_values_reversed=np.exp(1j*shi_1)  #converting into complex form
    
    return steering_angle,steering_angle_rad,complex_values_reversed

def process_frames(file_path,complex_values_reversed,n_avg,n_avg_sub):
    frame_size=8330  #no. of bytes per frame including frame headers
    init_skip=start_index*frame_size*100  #initial skip
    freq_bins=4096  #freq bins
    batch_size=100  #reading 100 frames as a batch every time for GPU parallel processing
    
    n_frame=(stop_index-start_index)*100   
    spec_2d = cp.zeros([freq_bins,n_frame//n_avg])  #spec_2d dummy array
    
    accumulated_spec=cp.zeros([freq_bins,n_avg],dtype=cp.complex128)  #accumulated_spec dummy array
    frame_counter=0
    f = open(file_path, 'rb')
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
            batch_data = cp.zeros((batch_size, 8192), dtype=cp.uint8)   #creating dummy array for batch data
            for i in range(batch_size):  #read batch_size times in loop and create a 2d array as (100,8192)
                f.seek(138,1)  #skipping 138 file headers
                batch_data[i,:]=cp.asarray(cp.frombuffer(f.read(8192), dtype=cp.uint8))  #reading 8192 bytes and storing in batch_data

            ch_data=ref_data[batch_data,:8]   #using LUT reference converting into 1s and -1s
            
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
            channel_data_bins=channel_data[:,:,freq_bins:]

            # reshaping the time delay array by adding one new axis means converting from (8,4096) to (8,dummy,4096)
            complex_values_reversed_reshaped = cp.tile(complex_values_reversed[:, cp.newaxis, :], ( 1,100, 1))

            #multiplying channel data bins with time delay 
            channel_data_complex=channel_data_bins*cp.asarray(complex_values_reversed_reshaped)

            #summing the 8 channels after multipying with time delay
            accumulated_spec[:,:]= cp.sum(channel_data_complex, axis=0).T

            #averaging every 100 frames and making as 1 spectrum
            
            avg_spectra = cp.mean(cp.abs(accumulated_spec), axis=1)
            spec_2d[:, frame//n_avg] = cp.array(avg_spectra)
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
    
    
    spec_2d_db = 10 * np.log10(spec_2d.get()) +1e-10 #1e-10 to handle divide by zero warnings
    
    #spec_2d=np.power(10,spec_2d_db/10) #reverse 10log10
    # Save each frame (as an image) to the frames list

    avg_first_100_spectrums_abs = np.median(spec_2d_db[:, :n_avg_sub], axis=1) #  for first 100 bg sub (:n_avg_sub means first 100)
    #avg_first_100_spectrums_abs = np.median(spec_2d_db[:, -n_avg_sub:], axis=1)  #for last 100 bg sub (-n_avg_sub:  means last 100)
    spec_2d_abs_sub = spec_2d_db - avg_first_100_spectrums_abs[:, np.newaxis]  #bg sub
    
    
    return spec_2d,spec_2d_abs_sub

   
    
def save_spectra_plot(spec_2d_abs_sub,output_file_path,steering_angle_rad,fmin,fmax,freq_axis,freq_bin_axis,time_axis,time_bin_axis,actual_steering_angle,title_name,start_time_file):
    
    #--------------------------Beam pattern--------------------------------
        
    # Beam pattern calculation
    d_spac= 5  
    theta = np.arange(-180, 180, 0.1) 
    mu=0
    hpbw_theta = 80;
    sigma_e = hpbw_theta / 2.355;
    y_single  = (1/(sigma_e*np.sqrt(2*np.pi))) * np.exp(-0.5*((theta - mu)/sigma_e)**2)
    
    def calculate_beam_pattern(freq):
        
         
        theta_rad = np.deg2rad(theta)  
        num=8
        c=3e8
         
        lam = c / freq  
    
        
        array_positions = np.arange(num)
        k = 2 * np.pi / lam  # Wavenumber
        delta = k * d_spac * (np.sin(theta_rad[:, np.newaxis]) - np.sin(steering_angle_rad))
        arrayfactor_tot = np.sum(np.exp(1j * array_positions * delta), axis=1)    #Formula beamforming
        #arrayfactor_scaled=arrayfactor_tot*y_single
        #arrayfactor_norm = arrayfactor_scaled / np.max(arrayfactor_scaled)
        
        AF_norm = abs(arrayfactor_tot / max(arrayfactor_tot))**2  # Normalize array factor
       
        # Scale by single antenna beam pattern
        gain = AF_norm * y_single / max(y_single)  # af_scaled
        
        #gain = np.abs(arrayfactor_norm)**2
        return gain,lam
    #----------------------------------------------------------------------
    
    #col avg and row avg
    col_summed_spectrum =np.nanmean(np.nan_to_num(spec_2d_abs_sub, nan=np.nan, posinf=np.nan, neginf=np.nan),axis=0)
    row_summed_spectrum = np.mean(spec_2d_abs_sub, axis=1)   # Sum over rows
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0,10,3], width_ratios=[3, 0.0, 1])
    
    #row avg - axis1
    ax1 = fig.add_subplot(gs[1, 2])
    ax1.plot(row_summed_spectrum[fmin:fmax] , freq_bin_axis[fmin:fmax])  #freq_axis[fmin:fmax]
    ax1.set_xlabel('db above bg',fontsize=14)
    ax1.set_ylabel('Frequency Bins',fontsize=14)
    ax1.set_xlim(-0.5,1.5)
    ax1.set_ylim(fmin,fmax)
    # ax1.set_xlabel('Power')
    # ax1.set_title('Summed by Row')
    ax1.grid()
    ax1.yaxis.tick_right()  # marking at right side
    ax1.yaxis.set_label_position("right")  
    #ax1.set_ylabel('Frequency (MHz)')  
    
    #spectrum plot ax2
    ax2 = fig.add_subplot(gs[1, 0])
    plt.imshow(spec_2d_abs_sub[fmin:fmax,:], aspect='auto', extent=[time_axis.min(), time_axis.max(), freq_axis[fmin], freq_axis[fmax-1]], origin='lower', cmap='jet',vmin=v_min,vmax=v_max)
    #plt.xticks(visible=False)  # to hide x axis on this plot
    #bar = plt.colorbar(im,pad=0.15,location='left') 
    #im.xaxis.tick_top()
    #im.xaxis.set_label_position("top")
    ax2.set_xticks(np.linspace(time_axis.min(), time_axis.max(), 5))
    ax2.set_xlabel('Time (UT - Decimal Hrs)')
    ax2.set_ylabel('Frequency (MHz)',fontsize=14)
    #ax2.set_title(f'{title_name}')
    
    '''ax2_1 = ax2.twinx()  
    ax2_1.set_ylim(freq_bin_axis.min(),freq_bin_axis.max())
    ax2_1.set_ylabel('Frequency Bins')'''
    
    # Summed over Columns ax3
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time_axis,col_summed_spectrum)
    ax3.set_xticks(np.linspace(time_axis.min(), time_axis.max(), 5))
    ax3.set_xlim(time_axis.min(),time_axis.max())
    #ax3.set_ylim(0,2)
    ax3.grid()
    #ax3.axis('tight')
    ax3.set_xlabel('Time (UT - Decimal Hrs)',fontsize=14)
    ax3.set_ylabel('db above bg',fontsize=14)
    
    

    #beam representaion -ax4
    ax4 = plt.subplot(gs[2, 2])  # Beam pattern plot (bottom-right corner)
    #ax4.plot(theta, np.abs(arrayfactor_norm)**2, 'blue', linewidth=2)  # Plot beam pattern
    
    # ----------------------------- shading of main and grating lobes-----------------------------
    def angle_from_sin(sin_val):
        if -1 <= sin_val <= 1:
            return np.rad2deg(np.arcsin(sin_val))
        return None
    colors = ['b', 'r']
    ax4.plot(theta,y_single/max(y_single),linewidth=1,linestyle='dashed')  # plotting single antenna
    freqs = [50e6,70e6]  
    for i in range(len(freqs)):
        freq = freqs[i]
        gain, lam = calculate_beam_pattern(freq)
        #label = f'Beam Pattern {freq/1e6:.0f} MHz'
        ax4.plot(theta, gain, colors[i], linewidth=2,label=f'{freq/1e6} MHz')
        
        # shading of main and grating lobes
        n_range = range(0, 2)  # n=0 for main lobes, n>0 for grating lobes
        for n in n_range:
            sin_theta_lobe = np.sin(steering_angle_rad) + n * lam / d_spac    #formula for lobe appear
            lobe_angle = angle_from_sin(sin_theta_lobe)
           
            #angle lies bw -90to90
            if lobe_angle is not None and -180 <= lobe_angle <= 180:
                center = lobe_angle  # center of lobe
                width = 15  # shading width of lobe
               
                #in given specfic region the shades will be plotted
                mask = (theta >= center - width/2) & (theta <= center + width/2)
                color = 'green' if np.isclose(center, steering_angle, atol=1) else 'blue'
                hatch = '+++' if color == 'green' else '***'
                    
                ax4.fill_between(theta[mask], gain[mask], color=color, alpha=0.3,
                                 hatch=hatch, edgecolor='k')
                
        
        # ploting the source for every freq
        ax4.scatter(actual_steering_angle, 1, color='orange', s=100)
    
    ax4.set_xlabel('Declination (in degrees)',fontsize=14)
    ax4.set_ylabel('Normalized Power',fontsize=14)
    ax4.legend()
    #ax4.set_title(f'Beam Pattern (Freq: {freq_beam/1e6} MHz) at {steering_angle:.2f}°')
    ax4.grid(True)
    
    

    end_time_ang = time.time()
    execution_time_ang = end_time_ang - start_time_ang
        
    # adjusting grid shap
    plt.suptitle(f'{title_name}_Execution_Time_{round(execution_time_ang,2)}_secs',fontsize=16 )
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.savefig(output_file_path,bbox_inches='tight')
    
    #plt.axis('tight')
    
    plt.show() 
    plt.close()

if __name__ == "__main__":
    start_time_exe=time.time()  #starting execution time 
    
    file_paths=select_files()   #fn calling to select file 
    
    for file_path in file_paths:  #loop for multiple files
        
        cp.get_default_memory_pool().free_all_blocks()   #command to free gpu memory
        #splitting the filename from filepath
        file_name = os.path.basename(file_path).split('.')[0] + '_' + os.path.basename(file_path).split('.')[1]
        
        Decl=sun_dec(file_name) #fn calling to find sun's declination
        angles = [Decl]  #angles
        
        
        actual_decl=Decl  #actual decl
        actual_steering_angle=Decl-13.6  #steer angle position
        for Decl in angles:
            cp.get_default_memory_pool().free_all_blocks()  #command to free gpu memory
            
            start_time_ang = time.time()
            print(f'Suns Declination: {Decl}')
            steering_angle,steering_angle_rad,complex_values_reversed=time_delay(Decl)
            

            
            
            #for full file edit t_start and t_end at calculate_time_axis function
            #at time_axis function t_start and t_end = min and max for full file
            #if full file not needed then make it as command
            
            # enter sun file diff means if first file sun number is 2 means then 2-1=1 then enter sun file diff as 1
            #if sun number in file and starting file has same number means enter 0
            # enter diff of sun start value and normal count value on sun_file_diff
            #enter sun_file_diff as starting file-1
            
            
            #sun_file_diff=1  # on 11.3.25
            #sun_file_diff=0  # on 12.3.25
            #sun_file_diff= 0  # on 13.3.25
            #sun_file_diff=0  # on 14.3.25
            #sun_file_diff=3  # on 15.3.25
            #sun_file_diff=3  # on 15.3.25
            #sun_file_diff=267  # on 18.3.25
            #sun_file_diff=833  # on 23.3.25
            #sun_file_diff=1487  # on 27.3.25
            #sun_file_diff=2003  # on 30.3.25
            #sun_file_diff=2168  # on 31.3.25
            #sun_file_diff=2452  # on 2.4.25
            #sun_file_diff=2598  # on 3.4.25
            #sun_file_diff=2843  # on 4.4.25
            #sun_file_diff=2976  # on 5.4.25
            #sun_file_diff=3126  # on 6.4.25
            #sun_file_diff=3260  # on 7.4.25
            #sun_file_diff=3400  # on 8.4.25
            #sun_file_diff=4623 # on 17.04.2025
            #sun_file_diff=4841 # on 18.04.2025
            #sun_file_diff=5210 # on 20.04.2025
            #sun_file_diff=5508 # on 21.04.2025.
            #sun_file_diff=5703 # on 22.04.2025
            #sun_file_diff=5983 # on 24.04.2025
            #sun_file_diff=6132 # on 25.04.2025 for first half files
            #sun_file_diff=6176 # on 25.04.2025 for 2nd half files
            #sun_file_diff=6208 # on 25.04.2025 for 3rd half files
            #sun_file_diff=6298 # on 26.04.2025
            #sun_file_diff=6457 # on 27.04.2025
            #sun_file_diff=6616 # on 28.04.2025
            #sun_file_diff=6927 # on 29.04.2025
            #sun_file_diff=7090 # on 30.04.2025
            #sun_file_diff=7310 # on 1.5.2025
            #sun_file_diff=7543 # on 02.5.2025
            #sun_file_diff=7725 # on 03.5.2025
            #sun_file_diff=7873 # on 04.5.2025
            sun_file_diff=8008 # on 05.5.2025
            
            #enter color scale vmin and vmax
            
            v_min= 0
            v_max= 0.5
            
            #fn calling to find time axis
            t_axis=calculate_time_axis(file_name, sun_file_diff)
            

            #for full file
            t_start=t_axis.min()
            t_end=t_axis.max()
            
            #finding starting and stopping index
            diff_start=abs(t_axis-t_start)
            index_start=diff_start.argmin()
            
            diff_end=abs(t_axis-t_end)
            index_end=diff_end.argmin()
            
            #spec avg and bg sub 
            n_avg=100
            n_avg_sub=100
            
            
            #------finding start and stop using start and end time index-----------
            start_index=index_start//100
            stop_index=index_end//100
            no_frames=stop_index-start_index
            print(f'{no_frames} Loops going to run')
            #----------------------------------------------------------------------
            
            #all variables sending to process frames are in global scope. So no need to send all. Directly it will take
            spec_2d,spec_2d_abs_sub=process_frames(file_path, complex_values_reversed,n_avg,n_avg_sub)
            
            fmin=0
            fmax=4096
            freq_bins=4096
            
            #output folder declaration
            output_folder = '/home/pulsar/Documents/intern_work/Observation_init_skip_images_time_axis_GPU/2025_05_05/images'
            output_folder_bin = '/home/pulsar/Documents/intern_work/Observation_init_skip_images_time_axis_GPU/2025_05_05/bin_file'
            
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(output_folder_bin, exist_ok=True)
            
            #title declaration
            title_name = (file_name + f'_start_time_{index_start}_end_time_{index_end}_{n_avg}_points_avg(bg_sub_{n_avg_sub}_pts)\n'
                          f'(sky_position_{round(Decl,2)}°_steer_angle_{round(steering_angle,2)}°)_({no_frames}_pts_plot)_({start_index}_initial_skip)_(actual_delay)_(mean_after_abs)_10log10_plot_vmin={v_min},vmax={v_max}\n'
                          f'plot_actual_Decl_{round(actual_decl,2)}°_GPU_first_100_bgsub')
            #output name delaration
            output_name=file_name + f'_(sky_position_{round(Decl,2)}°_steer_angle_{round(steering_angle,2)}°)_{no_frames}_loops_vmin={v_min},vmax={v_max}'
            
            
            output_file_path = os.path.join(output_folder, f'{output_name}.jpg')
            
            #freq axis and time axis generation
            freq_axis=np.linspace(45,90,freq_bins)  #start freq, stop freq, freq bins
            time_axis=np.linspace(t_start,t_end,no_frames)
            freq_bin_axis=np.linspace(fmin,fmax,freq_bins)
            time_bin_axis=np.linspace(start_index,stop_index,no_frames)
            
            #fn calling for plotting
            save_spectra_plot(spec_2d_abs_sub, output_file_path, steering_angle_rad, fmin, fmax, freq_axis, freq_bin_axis, time_axis, time_bin_axis, actual_steering_angle, title_name,start_time_ang)
            
            #-------------------WRITING TO BIN FILE--------------------------------
            # writing spec_2d needs to take bg sub and 10log10 after reading bin file
            array_size=np.size(spec_2d)
            os.makedirs(output_folder_bin, exist_ok=True)
            binary_output_file_path = os.path.join(output_folder_bin, f'{output_name}_{array_size}_array_size_float32.bin')
            spec_2d[fmin:fmax,:].astype(np.float32).tofile(binary_output_file_path)
            #----------------------------------------------------------------------

            
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