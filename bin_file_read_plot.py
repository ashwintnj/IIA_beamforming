import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import time
import sys
import ephem
from matplotlib.gridspec import GridSpec
import re



start_time_exe = time.time()  # start time of file

def select_files():
    # Create the root window for the file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select multiple .bin files
    return filedialog.askopenfilenames(
        title="Select .bin files", 
        filetypes=[("BIN files", "*.bin")]
    )
    
    
def calculate_time_axis(file_name,sun_file_diff):
    #------------TIME AXIS IN DECIMAL HRS CALCULATION------------------------------

    # enter sun file diff means if first file sun number is 2 means then 2-1=1 then enter sun file diff as 1
    #if sun number in file and starting file has same number means enter 0
    file_time=0.048  #2.88 min in hrs
    # enter diff of sun start value and normal count value on sun_file_diff
    #sun_file_diff=267
 

    time_str = file_name.split("_")[1][8:14]

    t_hh=int(time_str[:2])
    t_mm=int(time_str[2:4])
    t_ss=int(time_str[4:6])
    
    file_start=t_hh+(t_mm/60)+(t_ss/3600)
    

    #file_no= int(file_name[-len(str(sun_file_diff)):])-1-sun_file_diff  # change 4 or 3
    file_no=int(file_name.split("_")[5].replace('SUN',''))-1-sun_file_diff
    file_size=1900000
    start_time=file_start+(file_time*file_no)
    end_time=file_start+(file_time*(file_no+1))
    t_axis= np.linspace(start_time,end_time,file_size)
    
    
    return t_axis

def sun_dec(file_name):
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



def process_frames(file_path,file_array_size,n_avg_sub):
    
    
    spec_2d = np.fromfile(file_path,dtype=np.float32)  #change the data type of bin file if needed
    spec_2d = spec_2d.reshape(4096, file_array_size)
    
    spec_2d_db = 10 * np.log10(spec_2d) +1e-10 #1e-10 to handle divide by zero warnings
    
    #spec_2d=np.power(10,spec_2d_db/10) #reverse 10log10
    # Save each frame (as an image) to the frames list

    avg_first_100_spectrums_abs = np.median(spec_2d_db[:, :n_avg_sub], axis=1) #:n_avg_sub  means first 100 avg
    #avg_first_100_spectrums_abs = np.median(spec_2d_db[:, 2000:2100], axis=1) #-n_avg_sub:  means last 100 avg
    spec_2d_abs_sub = spec_2d_db - avg_first_100_spectrums_abs[:, np.newaxis]  #bg sub
    
    
    
    return spec_2d,spec_2d_abs_sub

   
    
def save_spectra_plot(spec_2d_abs_sub,output_file_path,steering_angle_rad,fmin,fmax,freq_axis,freq_bin_axis,time_axis,actual_steering_angle,title_name):
    
    #--------------------------Beam pattern--------------------------------
        
    # Beam pattern calculation
    d_spac= 5  
    theta = np.arange(-180, 180, 0.1) 
    #mu = np.rad2deg(steering_angle_rad);
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
        arrayfactor_tot = np.sum(np.exp(1j * array_positions * delta), axis=1)
        #arrayfactor_scaled=arrayfactor_tot*y_single
        #arrayfactor_norm = arrayfactor_scaled / np.max(arrayfactor_scaled)
        
        AF_norm = abs(arrayfactor_tot / max(arrayfactor_tot))**2  # Normalize array factor
       
        # Scale by single antenna beam pattern
        gain = AF_norm * y_single / max(y_single)  # af_scaled
        
        #gain = np.abs(arrayfactor_norm)**2
        return gain,lam
    #----------------------------------------------------------------------
    
        
 
    #spec_2d_abs_sub=spec_2d
    # 1d spect mult
    #col_summed_spectrum = np.sum(spec_2d, axis=0)  # Sum over columns
    col_summed_spectrum =np.nanmean(np.nan_to_num(spec_2d_abs_sub, nan=np.nan, posinf=np.nan, neginf=np.nan),axis=0)
    row_summed_spectrum = np.mean(spec_2d_abs_sub, axis=1)   # Sum over rows
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0,10,3], width_ratios=[3, 0.0, 1])
    
    
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
    
    # 1D Spectrum (Summed over Columns) - Bottom
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time_axis,col_summed_spectrum)
    ax3.set_xticks(np.linspace(time_axis.min(), time_axis.max(), 5))
    ax3.set_xlim(time_axis.min(),time_axis.max())
    ax3.set_ylim(-0.5,2)
    ax3.grid()
    #ax3.axis('tight')
    ax3.set_xlabel('Time (UT - Decimal Hrs)',fontsize=14)
    ax3.set_ylabel('db above bg',fontsize=14)
    
    

    
    ax4 = plt.subplot(gs[2, 2])  # Beam pattern plot (bottom-right corner)
    #ax4.plot(theta, np.abs(arrayfactor_norm)**2, 'blue', linewidth=2)  # Plot beam pattern
    
    # ----------------------------- shading of main and grating lobes-----------------------------
    def angle_from_sin(sin_val):
        if -1 <= sin_val <= 1:
            return np.rad2deg(np.arcsin(sin_val))
        return None
    colors = ['b', 'r']
    ax4.plot(theta,y_single/max(y_single),linewidth=1,linestyle='dashed')
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
                #label = 'Main Lobe' if color == 'green' else 'Grating Lobe'
    
                ax4.fill_between(theta[mask], gain[mask], color=color, alpha=0.3,
                                 hatch=hatch, edgecolor='k')
                
        
        # Plot the source point for each frequency
        ax4.scatter(actual_steering_angle, 1, color='orange', s=100)
    
    ax4.set_xlabel('Declination (in degrees)',fontsize=14)
    ax4.set_ylabel('Normalized Power',fontsize=14)
    ax4.legend()
    #ax4.set_title(f'Beam Pattern (Freq: {freq_beam/1e6} MHz) at {steering_angle:.2f}°')
    ax4.grid(True)
    


        
    # adjusting grid shap
    plt.suptitle(f'{title_name}',fontsize=16 )
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.savefig(output_file_path,bbox_inches='tight')
    
    #plt.axis('tight')
    
    plt.show() 
    #plt.close()

if __name__ == "__main__":
    start_time_exe=time.time()
    
    file_paths=select_files()

    n_avg_sub=100
    file_array_size=18999  #change here with respect to file size
    
    print(f'{len(file_paths)} plots needs to generate')
    i=0
    for file_path in file_paths:
        
        i=i+1
        print(i)
        #file_name = os.path.basename(file_path).split('.')[0] + '_' + os.path.basename(file_path).split('.')[1]
        file_name = os.path.basename(file_path)

        Decl=sun_dec(file_name)
        #t_start=6.4225 #for 18.3.25 event 
        #t_end=6.4276  # for 18.3.25 event
        
        #for full file edit t_start and t_end at calculate_time_axis function
        #at time_axis function t_start and t_end = min and max for full file
        #if full file not needed then make it as command
        
        # enter sun file diff means if first file sun number is 2 means then 2-1=1 then enter sun file diff as 1
        #if sun number in file and starting file has same number means enter 0
        # enter diff of sun start value and normal count value on sun_file_diff
        #enter sun_file_diff as starting file-1
        
        #sun_file_diff=0  # on 13.3.25
        #sun_file_diff=0  # on 14.3.25
        #sun_file_diff=3  # on 15.3.25
        #sun_file_diff=267  # on 18.3.25
        #sun_file_diff=833  # on 23.3.25
        #sun_file_diff=1487  # on 27.3.25
        #sun_file_diff=2003  # on 30.3.25
        #sun_file_diff=2168  # on 31.3.25
        sun_file_diff=2452  # on 2.4.25
        #sun_file_diff=2598  # on 3.4.25
        #sun_file_diff=2843  # on 4.4.25
        #sun_file_diff=2976  # on 5.4.25
        #sun_file_diff=3126  # on 6.4.25
        #sun_file_diff=3260  # on 7.4.25
        #sun_file_diff=3400  # on 8.4.25
        #sun_file_diff=6457 # on 27.04.2025
        #sun_file_diff=7090 # on 30.04.2025
        
        #enter color scale vmin and vmax
        v_min=0
        v_max=2
        
        t_axis=calculate_time_axis(file_name, sun_file_diff)
        

        #for full file
        t_start=t_axis.min()
        t_end=t_axis.max()
        
        diff_start=abs(t_axis-t_start)
        index_start=diff_start.argmin()
        
        diff_end=abs(t_axis-t_end)
        index_end=diff_end.argmin()
        

        
        
        #all variables sending to process frames are in global scope. So no need to send all. Directly it will take
        spec_2d,spec_2d_abs_sub=process_frames(file_path,file_array_size,n_avg_sub)
        
        fmin=0
        fmax=4096
        freq_bins=4096    
        
        match = re.search(r'sky_position_([-+]?\d*\.\d+|\d+)', file_name)
        
        if match:
            sky_position = match.group(1)
             
        array_lat=13.6
        
        steering_angle = float(sky_position) - array_lat
        steering_angle_rad = np.deg2rad(steering_angle)
        
        actual_steering_angle=Decl - array_lat
            
        
        #output_folder = '/home/pulsar/Documents/intern_work/Observation_init_skip_images_time_axis_GPU/2025_04_30/images'
        #output folder for my laptop
        output_folder = 'F:/IIA/02_04_2025_event_data_plots/test/'
        #output_folder_bin = '/home/pulsar/Documents/intern_work/Observation_init_skip_images_time_axis_GPU/2025_04_08/bin_file'
        
        os.makedirs(output_folder, exist_ok=True)
        #os.makedirs(output_folder_bin, exist_ok=True)
        
        first_line='_'.join(os.path.basename(file_path).split('_')[0:6])
        second_line='_'.join(os.path.basename(file_path).split('_')[6:14])  #[6:14] for new type of file name which is short [6:18] is for lengthy file name which is old type file name
        third_line= f'first_100_bg_sub_vmin={v_min},vmax={v_max}'
        #third_line='_'.join(os.path.basename(file_path).split('_')[14:20]).replace('.bin','')
        title_name = first_line +'\n'+second_line +'\n' +third_line 

        output_name=file_name.replace('.bin','') 
        
        #without bg sub only 10log10 title and output name
        #title_name = (file_name + f'_start_time_{index_start}_end_time_{index_end}_{n_avg}_points_avg_(bg_sub_{0}_pts)\n'
        #              f'(sky_position_{round(Decl,2)}°_steer_angle_{round(steering_angle,2)}°)_({no_frames}_pts_plot)_({start_index}_initial_skip)_(actual_delay)_(mean_after_abs)_10log10_plot_vmax_auto\n'
        #              f'plot_actual_Decl_{round(actual_decl,2)}°_GPU')
        
        #output_name=file_name + f'_start_time_{index_start}_end_time_{index_end}_(sky_position_{round(Decl,2)}°_steer_angle_{round(steering_angle,2)}°)_{no_frames}_loops_vmax_auto_10log10_plot'
        
        
        output_file_path = os.path.join(output_folder, f'{output_name}.jpg')
        
        
        freq_axis=np.linspace(45,90,freq_bins)
        time_axis=np.linspace(t_start,t_end,file_array_size)
        freq_bin_axis=np.linspace(fmin,fmax,freq_bins)
        #time_bin_axis=np.linspace(start_index,stop_index,file_array_size)
        
        save_spectra_plot(spec_2d_abs_sub, output_file_path, steering_angle_rad, fmin, fmax, freq_axis, freq_bin_axis, time_axis, actual_steering_angle, title_name)
        

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