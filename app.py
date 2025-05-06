"""
Created on Thu Jun  1 10:39:29 2023

@author: Anuruddha
"""
import numpy as np
import pickle 
import  streamlit as st
from streamlit_option_menu import option_menu
import lightkurve as lk
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt


# functions
def plot(x,y):
    fg,ax = plt.subplots(1,1)
    ax.plot(x,y)
    ax.set_title(f'The light curve of {KIC_No}')
    ax.set_xlabel("time (days)")
    ax.set_ylabel("flux ($\mathregular{es^{-1}}$)")
    fg.tight_layout()
    st.pyplot(fg)
    
    
def Chi_square(flux,err):
    
    mean = np.mean(flux)   
    chi_square = np.sum((flux-mean)**2/err**2)
    
    return chi_square



def Robust_median_statistic(flux,err):
    
    N = len(flux)
    median = np.median(flux)
    
    ROMS = (np.sum(abs(flux - median)/err))/(N-1)
    
    return ROMS



def Von_Neumann_ratio(flux):
    
    N = len(flux)
    mean = np.mean(flux)
    var = np.sum((flux - mean)**2/(N-1))
    delta_sq = 0
    
    for i in range(len(flux) - 1):
        
        flux_1 = flux[i]
        flux_2 = flux[i+1]
        
        delta_sq +=(flux_2 - flux_1)**2/(N-1)
    
    VNR = delta_sq/var
    
    return VNR
        
             

def Slop_Vector(time,flux):
    
    """calculate slope between two points. slope = distence_between_two_flux_points/distance_between_two_time_points
    retuens slope vector which contain slope between every two points."""
    
    flux_diff_vec = []
    time_diff_vec = []

    for i in range(len(flux)-1):
        
        flux_diff = flux[i+1] - flux[i]
        time_diff = time[i+1] - time[i]
    
        flux_diff_vec.append(flux_diff)
        time_diff_vec.append(time_diff)
    

    flux_diff_vec = np.array(flux_diff_vec)
    time_diff_vec = np.array(time_diff_vec)
    
    slope_vec = flux_diff_vec/time_diff_vec
    
    return slope_vec


def Peak_to_peak_variability(flux,err):
    
    fulx_min_sigma = []
    fulx_pls_sigma = []
    
    for i in range(len(flux)):
        
        f_m_s = flux[i] - err[i]
        f_p_s = flux[i] + err[i]
        
        fulx_min_sigma.append(f_m_s)
        fulx_pls_sigma.append(f_p_s)
        
    max_fms = max(fulx_min_sigma)
    min_fps = min(fulx_pls_sigma)
    
    v = (max_fms - min_fps)/(max_fms + min_fps)
    
    return v
    

def Lag_1_autocorrelation(flux):
    
    mean = np.mean(flux)
    numa = 0
    dino = np.sum((flux - mean)**2)
    
    for i in range(1,len(flux)):
        
        numa += (flux[i] - mean)*(flux[i-1] - mean)
    
    L1AC = numa/dino
    
    return L1AC



def fourier_comp2(time,flux):
    
    N = len(flux)
    T = time[1] - time[0]

    yf = np.fft.fft(flux)
    xf = np.fft.fftfreq(N,T)[1:N//2]
    
    max_freq_index = np.argwhere((np.abs(yf[1:N//2]) == np.max(np.abs(yf[1:N//2]))))
    max_freq = xf[max_freq_index]
    f1 = max_freq[0][0] 
    
    second_max_freq_index = np.argwhere((np.abs(yf[1:N//2])==np.sort((np.abs(yf[1:N//2])))[::-1][1]))
    second_max_freq = xf[second_max_freq_index]
    f2 = second_max_freq[0][0]
      
    return f1,f2


def fourier_comp(time,flux):
    
    N = len(flux)
    T = time[1] - time[0]

    yf = np.fft.fft(flux)
    xf = np.fft.fftfreq(N,T)[1:N//2]
    
    max_freq_index = np.argwhere((np.abs(yf[1:N//2]) == np.max(np.abs(yf[1:N//2]))))
    max_freq = xf[max_freq_index]
    f1 = max_freq[0][0] 
    
    return f1


def feature_genarate0(flux,err):
 
    flux_vec = flux[0]
    err_vec = err[0]

    # basic stat features
    median = np.median(flux_vec)
    
    std = np.std(flux_vec)  
    SSDM = np.sum((flux_vec-median)**2)
    MAD = np.median(abs(flux_vec - median))  
    ROMS = Robust_median_statistic(flux_vec,err_vec)
   
    
    feature_list = [std, SSDM, MAD, ROMS]

    return feature_list


def feature_genarate1(time,flux,err):
     
    feature_list = []

    for i in range(len(time)):
        
        features = []
    
        time_vec = time[i]
        flux_vec = flux[i]
        err_vec = err[i]
        
        # basic stat features 
        mean = np.mean(flux_vec)
        std = np.std(flux_vec)
        median = np.median(flux_vec)
        skew = stat.skew(flux_vec)
        kurtosis = stat.kurtosis(flux_vec)
        #mode = stat.mode(flux_vec)[0][0]
        mode = stat.mode(flux_vec.flatten(), keepdims=True).mode[0]
        SSDM = np.sum((flux_vec-median)**2)
        MAD = np.median(abs(flux_vec - median))
            
        # stat features (using functions)      
        slope_vec = Slop_Vector(time_vec,flux_vec)
        abs_slop_vec = abs(slope_vec)
        index = np.argmax(abs_slop_vec)

        Max_Slope = slope_vec[index]
        Abs_Max_Slope = abs(Max_Slope)
        Chi_sq = Chi_square(flux_vec,err_vec)
        ROMS = Robust_median_statistic(flux_vec,err_vec)
        VNR = Von_Neumann_ratio(flux_vec) 
        PTPV = Peak_to_peak_variability(flux_vec,err_vec)
        L1AC = Lag_1_autocorrelation(flux_vec)
             
        # append features to a list
        features.append(round(mean,4))
        features.append(round(std,4))
        features.append(round(median,4))
        features.append(round(skew,4))
        features.append(round(kurtosis,4))
        features.append(round(mode,4))
        features.append(round(SSDM,4))
        features.append(round(MAD,4))
             
        features.append(round(Max_Slope,4))
        features.append(round(Abs_Max_Slope,4))
        features.append(round(Chi_sq,4))
        features.append(round(ROMS,4))
        features.append(round(VNR,4))
        features.append(round(PTPV,4))
        features.append(round(L1AC,4))
  
        feature_list.append(features)
    
    return feature_list       


def feature_genarate2(time,flux,err):
    
    feature_list = []

    for i in range(len(time)):
        
        features = []
    
        time_vec = time[i]
        flux_vec = flux[i]
        err_vec = err[i]
        
        # basic stat features
        mean = np.mean(flux_vec)
        std = np.std(flux_vec)
        median = np.median(flux_vec)
        skew = stat.skew(flux_vec)
        kurtosis = stat.kurtosis(flux_vec)
        #mode = stat.mode(flux_vec)[0][0]
        mode = stat.mode(flux_vec.flatten(), keepdims=True).mode[0]
        SSDM = np.sum((flux_vec-median)**2)
        MAD = np.median(abs(flux_vec - median))
        
        # stat features (using functions)   
        slope_vec = Slop_Vector(time_vec,flux_vec)
        abs_slop_vec = abs(slope_vec)
        index = np.argmax(abs_slop_vec)

        Max_Slope = slope_vec[index]
        Abs_Max_Slope = abs(Max_Slope)
        Chi_sq = Chi_square(flux_vec,err_vec)
        ROMS = Robust_median_statistic(flux_vec,err_vec)
        VNR = Von_Neumann_ratio(flux_vec) 
        PTPV = Peak_to_peak_variability(flux_vec,err_vec)
        L1AC = Lag_1_autocorrelation(flux_vec)
        
        fc1,fc2 = fourier_comp2(time_vec,flux_vec)
          
        # append features to a list
        features.append(round(mean,4))
        features.append(round(std,4))
        features.append(round(median,4))
        features.append(round(skew,4))
        features.append(round(kurtosis,4))
        features.append(round(mode,4))
        features.append(round(SSDM,4))
        features.append(round(MAD,4))
              
        features.append(round(Max_Slope,4))
        features.append(round(Abs_Max_Slope,4))
        features.append(round(Chi_sq,4))
        features.append(round(ROMS,4))
        features.append(round(VNR,4))
        features.append(round(PTPV,4))
        features.append(round(L1AC,4))    
        features.append(round(fc1,4))
        features.append(round(fc2,4))
          
        feature_list.append(features)
    
    return feature_list



def feature_genarate3(time,flux,err):
      
    feature_list = []

    for i in range(len(time)):
        
        features = []
    
        time_vec = time[i]
        flux_vec = flux[i]
        err_vec = err[i]
        
        # basic stat features
        mean = np.mean(flux_vec)
        std = np.std(flux_vec)
        median = np.median(flux_vec)
        skew = stat.skew(flux_vec)
        kurtosis = stat.kurtosis(flux_vec)
        #mode = stat.mode(flux_vec)[0][0]
        mode = stat.mode(flux_vec.flatten(), keepdims=True).mode[0]
        SSDM = np.sum((flux_vec-median)**2)
        MAD = np.median(abs(flux_vec - median))
          
        # stat features (using functions)   
        slope_vec = Slop_Vector(time_vec,flux_vec)
        abs_slop_vec = abs(slope_vec)
        index = np.argmax(abs_slop_vec)

        Max_Slope = slope_vec[index]
        Abs_Max_Slope = abs(Max_Slope)
        Chi_sq = Chi_square(flux_vec,err_vec)
        ROMS = Robust_median_statistic(flux_vec,err_vec)
        VNR = Von_Neumann_ratio(flux_vec) 
        PTPV = Peak_to_peak_variability(flux_vec,err_vec)
        L1AC = Lag_1_autocorrelation(flux_vec)
        
        fc = fourier_comp(time_vec,flux_vec)
             
        # append features to a list
        features.append(round(mean,4))
        features.append(round(std,4))
        features.append(round(median,4))
        features.append(round(skew,4))
        features.append(round(kurtosis,4))
        features.append(round(mode,4))
        features.append(round(SSDM,4))
        features.append(round(MAD,4))    
        
        features.append(round(Max_Slope,4))
        features.append(round(Abs_Max_Slope,4))
        features.append(round(Chi_sq,4))
        features.append(round(ROMS,4))
        features.append(round(VNR,4))
        features.append(round(PTPV,4))
        features.append(round(L1AC,4))  
        features.append(round(fc,4))  
        
        feature_list.append(features)
    
    return feature_list


def download_lightcurves(kic_number):

    for i in range(2):
        try:
            lc  = lk.search_lightcurve(kic_number, author = 'Kepler', cadence = 'long', quarter=9).download()
            break
        except:
            pass
    
    if lc is None:
        #st.error(f"No light curve found for the selected target : KIC {kic_number}.")
        return [],[],[]

    # extracrt data 
    time_t = np.array(lc.time.to_value('jd'))
    time_t = time_t - time_t[0]
    flux_t = np.array(lc.flux.to_value())
    err_t = np.array(lc.flux_err.to_value())
    
    time_t = time_t.tolist()
    flux_t = flux_t.tolist()
    err_t = err_t.tolist()

    return time_t,flux_t,err_t



def preprocess_data(time_t,flux_t,err_t):
        
        # create data frame
        data = {'time':time_t , 'flux':flux_t, 'err':err_t}
        df = pd.DataFrame(data)
        
        # fill nans using backword (main) and forward( if last point NaN) fill methods.   
        new_df_back = df.fillna(method='bfill')
        new_df = new_df_back.fillna(method='ffill')
          
        # predict   
        time_p = np.array([new_df.time.tolist()])
        flux_p = np.array([new_df.flux.tolist()])
        err_p = np.array([new_df.err.tolist()])

        return time_p, flux_p, err_p


def get_non_variable_prediction(features):

    final_reult = '' # final result 
    features = features.reshape(1,-1)

    if c0_model.predict(features)[0] == 0:
        final_reult = "Non Variable"
    else:
        final_reult = "Variable"

    return final_reult


def get_main_prediction(features):

    final_reult = '' # final result
            
    if c1_model.predict(features)[0] == 0:
    
        point2 = np.array(feature_genarate2(time_p,flux_p,err_p))
        
        if c3_model.predict(point2)[0] == 0:
            final_reult = " Binary + Pulsation"
        elif c3_model.predict(point2)[0] == 1:
            final_reult = "Pure Binary"

    else:
        final_reult = "Pulsation"

    return final_reult


def get_pulsation_predictions(features):

    final_reult = '' # final result

    if c2_model.predict(features)[0] == 0:
        final_reult = "Delta Scuti"
    elif c2_model.predict(features)[0] == 1:
        final_reult = "Gamma Doradus"
    elif c2_model.predict(features)[0] == 2:
        final_reult = "RR Lyrae"
    elif c2_model.predict(features)[0] == 3:
        final_reult = " Solar-Like"

    return final_reult


def get_kic_list():

    input = st.text_input('KIC Number')
    if input == "":
        return []
    
    KIC_No_List = input.split(",")
    KIC_No_List = [kic for kic in KIC_No_List if kic != " "]
    return KIC_No_List


def get_inputs_from_upload(KIC_No_List):

    KIC_No_List_old = KIC_No_List

    uploaded_file = st.file_uploader("Choose a file", type=["txt", "dat"])  
    if uploaded_file is not None:
    # Display the file name
        st.write("Filename:", uploaded_file.name) 
        if uploaded_file.name.endswith(('.txt', '.dat')):
            # Read the TXT file
            text = uploaded_file.read().decode("utf-8")
            KIC_No_List = st.text_area("File Content", text, height=300, disabled=True).split("\r\n")
            return KIC_No_List
        
    return KIC_No_List_old

# loading saved models
c0_model = pickle.load(open("classifier_0.sav",'rb'))
c1_model = pickle.load(open("classifier_1.sav",'rb'))
c2_model = pickle.load(open("classifier_2.sav",'rb'))
c3_model = pickle.load(open("classifier_3.sav",'rb'))


# create side bar or navigation
with st.sidebar:
    
    selected = option_menu('Variable Star Classification System',
                           ['All Predictons', 'Main Prediction'],
                           
                           icons = ['house','stars'], # icons from bootstrap website
                           
                           default_index = 0) # default_index is defalut selected page


# All Prediction page
if (selected == 'All Predictons'):
    
    # page title
    st.title('Pulsation Types, Pure Binary and Binary + Pulsation Prediction')  

    text = ''
    KIC_No_List = []
    col1,col2 = st.columns(2)
    with col1:
        KIC_No_List = get_kic_list()
 
    KIC_No_List = get_inputs_from_upload(KIC_No_List)   

    col_names = ["KIC number", "Main Class", "Variable Class"]
    my_df  = pd.DataFrame(columns = col_names) 
    # creating a button for prediction 
    if st.button('Type of the star'):

        if len(KIC_No_List) == 0:
            st.error(f"Enter at leaset one KIC number")
            st.stop()

        if len(KIC_No_List) == 1:
            isPlot = True
        else:
            isPlot = False
  
        for i,KIC_No in enumerate(KIC_No_List):

            time_t, flux_t, err_t = download_lightcurves(KIC_No)
            if time_t == []:
                primary_reult = ""
                secondary_reult = ""
                st.error(f"No light curve found for the selected target: KIC {KIC_No}")
            else:
                time_p, flux_p, err_p = preprocess_data(time_t,flux_t,err_t)

                # find the given star is a variable or non-variable
                feature_set0 = np.array(feature_genarate0(flux_p,err_p))
                non_variable_reult = get_non_variable_prediction(feature_set0)

                if non_variable_reult=="Variable":
                    feature_set1 = np.array(feature_genarate1(time_p,flux_p,err_p))
                    primary_reult = get_main_prediction(feature_set1)

                    if primary_reult == "Pulsation":
                        feature_set2 = np.array(feature_genarate3(time_p,flux_p,err_p))
                        secondary_reult = get_pulsation_predictions(feature_set2)
                        st.success(f"KIC {KIC_No} is a {secondary_reult} {primary_reult} star") # display result 
                        if isPlot:   
                            plot(time_p[0],flux_p[0])
                    else:
                        secondary_reult = ""
                        st.success(f"KIC {KIC_No} is a {primary_reult} star") # display result 
                        if isPlot:   
                            plot(time_p[0],flux_p[0])
                else:
                    st.error(f"KIC {KIC_No} is a {non_variable_reult} star") # display result 
                    primary_reult = "Non Variable"
                    secondary_reult = ""

            my_df.loc[i] = [KIC_No, primary_reult, secondary_reult]

        my_df.index = my_df.index + 1
        my_df.index.name = "No"
        st.table(my_df)
        csv = my_df.to_csv().encode("utf-8")

        st.download_button(
        label="Download CSV",
        data=csv,
        file_name="all-prediction-results.csv",
        mime="text/csv",
        icon=":material/download:")

       
                
        
# Main Prediction page
if (selected == 'Main Prediction'):
    
    # page title
    st.title('Pulsation, Pure Binary and Binary + Pulsation Prediction')    
    text = ''
    KIC_No_List = []
    col1,col2 = st.columns(2)
    with col1:
         KIC_No_List = get_kic_list()
    
    KIC_No_List = get_inputs_from_upload(KIC_No_List)

    col_names = ["KIC number", "Main Class"]
    my_df  = pd.DataFrame(columns = col_names) 
    
    # creating a button for prediction 
    if st.button('Type of  the star'):

        if len(KIC_No_List) == 0:
            st.error(f"Enter at leaset one KIC number")
            st.stop()

        if len(KIC_No_List) == 1:
            isPlot = True
        else:
            isPlot = False
  
        for i,KIC_No in enumerate(KIC_No_List):

            time_t, flux_t, err_t = download_lightcurves(KIC_No)
            if time_t == []:
                primary_reult = ""
                st.error(f"No light curve found for the selected target: KIC {KIC_No}")
            else:
                time_p, flux_p, err_p = preprocess_data(time_t,flux_t,err_t)

                # find the given star is a variable or non-variable
                feature_set0 = np.array(feature_genarate0(flux_p,err_p))
                non_variable_reult = get_non_variable_prediction(feature_set0)

                if non_variable_reult=="Variable":
                    feature_set1 = np.array(feature_genarate1(time_p,flux_p,err_p))
                    primary_reult = get_main_prediction(feature_set1)

                    st.success(f"KIC {KIC_No} is a {primary_reult} star") # display result 
                    if isPlot:   
                        plot(time_p[0],flux_p[0])
                else:
                    st.error(f"KIC {KIC_No} is a {non_variable_reult} star") # display result 
                    primary_reult = "Non Variable"

            my_df.loc[i] = [KIC_No, primary_reult]

        my_df.index = my_df.index + 1
        my_df.index.name = "No"
        st.table(my_df)
        csv = my_df.to_csv().encode("utf-8")

        st.download_button(
        label="Download CSV",
        data=csv,
        file_name="main-prediction-results.csv",
        mime="text/csv",
        icon=":material/download:")
         
            