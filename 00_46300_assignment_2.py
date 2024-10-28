#%% Module imports
#General imports
import numpy as np
import pandas as pd
import cmath
import os
from intersect import intersection

#Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt

#%% Global plot settings

#Figure size:
mpl.rcParams['figure.figsize'] = (16, 8)  

#Lines and markers
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 7
mpl.rcParams['scatter.marker'] = "+"
mpl.rcParams['lines.color'] = "k"
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])
#Cycle through linestyles with color black instead of different colors
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])\
                                + mpl.cycler('linestyle', ['-', '--', '-.', ':'])\
                                + mpl.cycler('linewidth', [1.2, 1.2, 1.3, 1.8])

#Text sizes
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 20

#Padding
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20

#Latex font
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams["pgf.texsystem"] = "pdflatex"  # Use pdflatex for generating PDFs
mpl.rcParams["pgf.rcfonts"] = False  # Ignore Matplotlib's default font settings
mpl.rcParams['text.latex.preamble'] = "\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
                                                 r'\usepackage{siunitx}'])
mpl.rcParams.update({"pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])})

#Export
mpl.rcParams['savefig.bbox'] = "tight"

#%% Class definition
class WindFarmCalculator:
    def __init__(self, p=160, V_rtd=10, rpm_Rrtd=10, V_out = 25,
                 P_rtd=10e6, Psi=19.5, f_s=50, a=33/4,
                 exp_fld = "./00_export/"):
        # Assignment specifications
        self.p = p                            #[-] - Number of pole pairs
        self.V_rtd = V_rtd                    #[m/s] - Rated wind speed of the turbines
        self.V_out = V_out                    #[m/s] - Cut out wind speed
        omega_Rrtd = rpm_Rrtd*2*np.pi/60      #[rad/s] - Rated rotational of the turbine rotor
        self.omega_G_rtd = omega_Rrtd*p       #[rad/s] - Rated rotational speed of the electronic circuit
        self.P_rtd = P_rtd                    #[W] - Rated power of the turbines
        self.Psi = Psi                        #[Wb] - Rotor flux linkage
        self.omega = 2*np.pi*f_s              #[rad/s] - Rotational speed of the system-side circuit
        self.a = a                            #[-] - Turns ratio of the Transformer
        
        # Preparatory calculations
        self.Z_t1_p = a**2 * complex(25e-3, self.omega*10e-6)
        self.Z_t2 = self.Z_t1_p
        self.Z_t3_p = a**2 * complex(0, 121.4*self.omega*55e-3)/complex(121.4, self.omega*55e-3)
        self.Z_c1 = complex(8, self.omega*26e-3)
        self.Z_c2 = complex(0, -1/(self.omega*2e-6))
        
        self.calc_E_a = lambda omega_G: omega_G*self.Psi
        self.calc_omega_G = lambda V_0: self.omega_G_rtd*V_0/self.V_rtd
        
        self.A_1 = (1+(self.Z_t2+self.Z_c1)/self.Z_t3_p)
        self.B_1 = self.Z_t1_p+(self.Z_t2+self.Z_c1)*(self.Z_t1_p/self.Z_t3_p+1)
        
        self.A_i = (1/self.Z_c2)*(1+(self.Z_t2+self.Z_c1)/self.Z_t3_p) \
                    + 1/self.Z_t3_p
        self.B_i = 1 + (self.Z_t2+self.Z_c1)/self.Z_t3_p
        
        self.A_v = (self.Z_t1_p/self.Z_c2)*(1+(self.Z_t2+self.Z_c1)/self.Z_t3_p)\
                   + self.Z_t1_p/self.Z_t3_p \
                   + 1 + (self.Z_t2+self.Z_c1)/self.Z_t3_p
        self.B_v = self.Z_t1_p*(1+(self.Z_t2+self.Z_c1)/self.Z_t3_p) \
                   + self.Z_t2 + self.Z_c1
        
        #Misc
        self.exp_fld = exp_fld
        if not os.path.isdir(self.exp_fld):
            os.mkdir(self.exp_fld)
    
    def calc_P_mech(self, V_0):
        """Calculate the mechanical power of the rotor for a specified wind 
        speed.
        
        Parameters:
            V_0 (scalar or array-like);
                Wind speed [m/s]
        
        Returns:
            P_mech (scalar or array-like);
                The mechanical power for the specified wind speeds [W]
        """
        
        scalar_input = np.isscalar(V_0)
        if scalar_input:
            if V_0<=self.V_rtd:
                P_mech = self.P_rtd*np.power(V_0,3)/self.V_rtd**3
            elif V_0<=self.V_out:
                P_mech = self.P_rtd
            else:
                P_mech = 0
        else:
            P_mech = np.full(len(V_0), self.P_rtd)
            
            i_below_rtd = np.argwhere(V_0<self.V_rtd)
            i_above_out = np.argwhere(V_0>self.V_out)
            
            if i_below_rtd.size>0:
                P_mech[i_below_rtd] = self.P_rtd*np.power(V_0[i_below_rtd],3)\
                                      /self.V_rtd**3
            
            if i_above_out.size>0:
                P_mech[i_above_out] = 0
                
        return P_mech

    def calc_generator(self, V_0, cont_type="a", plt_res = False):
        """Calculate the electronic parameters of the generator circuit for a 
        specified wind speed and Maximum Power Point Tracking (MPPT) control 
        scheme.
        
        Parameters:
            V_0 (scalar or array-like);
                Wind speed [m/s]
            cont_type (str):
                MPPT control scheme. Available control schemes are:
                - "a": Zero Power Factor: The phase current is in phase with  
                       the terminal voltage
                - "b": Zero Beta Angle: The phase current is in phase with 
                       the induced voltage   
        
        Returns:
            V_vscg (scalar or array-like);
                The terminal voltage [V]
            I_a  (scalar or array-like);
                The current in the circuit [A]
            delta  (scalar or array-like);
                The phase angle between the terminal voltage and the 
                induced voltage [deg]
        """
        
        omega_G = self.calc_omega_G(V_0)
        E_a = self.calc_E_a(omega_G)
        
        R_s = 194.8e-3
        X_L = omega_G*1.8e-3
        
        if cont_type == "a":
            delta = .5*np.arcsin(self.calc_P_mech(V_0)*X_L
                                 * 2/3 / np.power(E_a,2))
            I_a = E_a/X_L*np.sin(delta)
            V_vscg = E_a*np.cos(delta)-I_a*R_s
        elif cont_type == "b":
            I_a = self.calc_P_mech(V_0)/(3*E_a)
            delta = np.arctan((I_a*X_L)/(E_a-I_a*R_s))
            V_vscg = E_a*np.cos(delta) \
                     - I_a*R_s*np.cos(delta) + I_a*X_L*np.sin(delta)
        else:
            raise ValueError(f"Invalid MPPT control scheme '{cont_type}'")
        
        i_0 = np.argwhere(V_0==0)
        if i_0.size>0:
            delta[i_0]=0
            I_a[i_0]=0
            V_vscg[i_0]=0
        
        if plt_res:
            self.plot_res(V_0, abs(np.vstack([V_vscg, E_a, I_a])), 
                          plt_labels=[r"$V_\text{VSC-G}$", r"$E_{a}$", 
                                      r"$I_{a}$"], 
                         ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$V\:[\unit{\V}]\text{ or }I\:[\unit{\A}]$"], 
                         ax_lims=[(),(0,3500)],
                         fname=f"Gen_circuit_MPPT{cont_type}")
        
        return V_vscg, I_a, delta
    
    def calc_system_circuit_t1(self, P_vscg, V_vscs_p):
        """Calculate the grid voltage and current based on the active power of 
        the generator-side voltage source converter (VSC) and an assumed value 
        of the voltage of the system side VSC.
        
        Parameters:
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC [W]
            V_vscs_p (scalar or array-like);
                Voltage of the system-side VSC, referred to the high voltage 
                side[V]
            
            
        Returns:
            V_poc (scalar or array-like);
                The power grid voltage [V]
            I_poc  (scalar or array-like);
                The power grid current [A]
            V_sec (scalar or array-like);
                Voltage of the secondary coil of the transformer (high 
                voltage side) [V]    
        """
        
        # if not type (V_vscs) == complex:
        #     if np.isscalar(V_vscs):
        #         V_vscs = complex(V_vscs, 0)
        #     else:
        #         V_vscs = np.array([complex(V, 0) for V in V_vscs])
        
        I_t1_p = P_vscg/(3*V_vscs_p)
        I_t3_p = (V_vscs_p-I_t1_p*self.Z_t1_p)/self.Z_t3_p
        I_t2 = I_t1_p-I_t3_p
        V_poc = I_t3_p*self.Z_t3_p-I_t2*(self.Z_t2+self.Z_c1)
        I_c2 = V_poc/self.Z_c2
        I_poc = I_t2-I_c2
        
        V_sec = V_poc + I_t2*self.Z_c1 #Voltage of the secondary coil of the transformer
        
        return V_poc, I_poc, V_sec
    
    def calc_system_circuit_t2(self, I_poc):
        """Calculate the grid voltage and current based on the active power of 
        the generator-side voltage source converter (VSC) and an assumed value 
        of the voltage of the system side VSC.
        
        Parameters:
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC [W]
            I_poc (scalar or array-like);
                The power grid current [A]
            
            
        Returns:
            V_vscs_p (scalar or array-like);
                Voltage of the system-side VSC, referred to the high voltage 
                side [V]
            I_t1_p (scalar or array-like);
                Current of the system-side VSC, referred to the high voltage 
                side [A]
            S_vscs (scalar or array-like);
                Apparent power of the system-side VSC [VA]  
            V_sec (scalar or array-like);
                Voltage of the secondary coil of the transformer (high 
                voltage side) [V]    
        """
        
        V_poc = 33e3/np.sqrt(3)
        I_c2 = V_poc/self.Z_c2
        I_t2 = I_c2 + I_poc
        I_t3_p = (I_t2*(self.Z_t2+self.Z_c1) + I_c2*self.Z_c2)/self.Z_t3_p
        I_t1_p = I_t2 + I_t3_p
        V_vscs_p = I_t1_p*self.Z_t1_p + I_t3_p*self.Z_t3_p
        
        V_sec = V_poc + I_t2*self.Z_c1 #Voltage of the secondary coil of the transformer
        
        S_vscs = 3*V_vscs_p*I_t1_p.conjugate()
        
        
        
        return V_vscs_p, I_t1_p, S_vscs, V_sec
        
    def solve_V_vscs_p (self, P_vscg):
        """Find the systems-side VSC voltage for which the power grid voltage 
        becomes equal to its nominal value of 33kV  for a specified active 
        power of the generator-side VSC assuming a power factor of 1 
        for the system-side VSC.
        
        Parameters:
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC [W]
        
        Returns:
            V_vscs (scalar or array-like);
                Voltage of the system-side VSC [V]
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC at which the solution 
                was found [W]
        """
        
        V_vscs_p_range = np.linspace(1,50e3, 500)
        
        V_poc = abs(V_vscs_p_range*self.A_1 
                    - P_vscg.reshape(-1,1)/3/V_vscs_p_range*self.B_1)

        V_vscs_p = np.zeros(len(P_vscg))
        V_poc_intsct = np.zeros(len(P_vscg))
        for i in range(len(P_vscg)):
            V_intersect = intersection(V_vscs_p_range, V_poc[i,:],
                                  [0,50e3], [33e3/np.sqrt(3), 33e3/np.sqrt(3)])
            
            V_vscs_p[i] = V_intersect[0][-1]
            V_poc_intsct[i] = V_intersect[1][-1]
            
        return V_vscs_p, V_poc_intsct 

    def solve_I_poc (self, P_vscg):
        """Find the systems-side VSC voltage and current for which the power 
        grid voltage becomes equal to its nominal value of 33kV  for a 
        specified active power of the generator-side VSC assuming a power 
        factor of 1 for the grid.
        
        Parameters:
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC [W]
        
        Returns:
            V_vscs_p (scalar or array-like);
                Voltage of the system-side VSC, referred to the high voltage 
                side [V]
            P_poc_intsct (scalar or array-like);
                Active power of the system-side VSC, at which the solution was 
                found [W]
        """
        I_poc_range = np.linspace(-100,500, 500)
        V_poc = 33e3/np.sqrt(3)
        
        S_vscs = 3*(V_poc*self.A_v + I_poc_range*self.B_v)\
                  *(V_poc*self.A_i.conjugate() 
                    + I_poc_range*self.B_i.conjugate())

        I_poc = np.zeros(len(P_vscg))
        P_poc_intsct = np.zeros(len(P_vscg))
        for i,P_i in enumerate(P_vscg):
            I_intersect = intersection(I_poc_range, S_vscs.real,
                                       [0,300], [P_i, P_i])
            
            try:
                I_poc[i] = I_intersect[0][-1]
                P_poc_intsct[i] = I_intersect[1][-1]
            except:
                pass
            
        return I_poc, P_poc_intsct 

    def calc_efficiency (self, V_0, cont_type="a", plt_res = False):  
        """Calculates the efficiency of the electronic system from the 
        generator to the power grid for specified wind speed(-s) and MPPT 
        control scheme
        
        Parameters:
            V_0 (scalar or array-like);
                Wind speed [m/s]
            cont_type (str):
                MPPT control scheme. Available control schemes are:
                - "a": Zero Power Factor: The phase current is in phase with  
                       the terminal voltage
                - "b": Zero Beta Angle: The phase current is in phase with 
                       the induced voltage  
        
        Returns:
            eta (scalar or array-like);
                The efficiency of the electronic system [-]
            P_poc (scalar or array-like);
                The active output power of the wind turbine to the grid [W]
            P_mech (scalar or array-like);
                The mechanical power for the specified wind speeds (i.e. the 
                input power into the system) [W]
        
        """
        P_mech = self.calc_P_mech (V_0)
        V_vscg, I_a, delta = self.calc_generator(V_0=V_0, cont_type=cont_type,
                                                 plt_res=plt_res)
        P_vscg = (3*V_vscg*I_a).real
        
        V_vscs_p, V_poc_intersect = self.solve_V_vscs_p(P_vscg=P_vscg)
        V_poc, I_poc, V_sec = self.calc_system_circuit_t1(P_vscg=P_vscg, 
                                                          V_vscs_p=V_vscs_p)
        
        P_poc = (3*V_poc*I_poc).real
        
        eta = P_poc/P_mech
        
        if plt_res:
            #Plot System-side circuit
            self.plot_res(V_0, abs(np.vstack([V_vscs_p, V_poc, V_sec]))*4/33, 
                          plt_labels=[r"$V_\text{VSC-S}$", r"$V_\text{POC}$", 
                                      r"$V_\text{sec}$"], 
                         ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$V\:[\unit{\V}]$"], 
                         ax_lims=[(),(2250,2500)],
                         fname=f"T1_Sys_circuit_lowVolt_MPPT{cont_type}")
            self.plot_res(V_0, abs(np.vstack([V_vscs_p, V_poc, V_sec])), 
                          plt_labels=[r"$V_\text{VSC-S}$", r"$V_\text{POC}$", 
                                      r"$V_\text{sec}$"], 
                         ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$V\:[\unit{\V}]$"], 
                         fname=f"T1_Sys_circuit_highVolt_MPPT{cont_type}")
            
            #Plot Power
            self.plot_res(V_0, np.vstack((P_mech, P_vscg, P_poc))/1e6, 
                          plt_labels=[r"$P_{mech}$", r"$P_\text{VSC-G}$",
                                      r"$P_{POC}$"], 
                          ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$P\:[\unit{\MW}]$"], 
                          ax_ticks=[[],np.arange(0,11,2)], 
                          fname=f"T1_P_vs_V0_MPPT{cont_type}")
            
            #Plot eta
            self.plot_res(V_0, eta*100, plt_labels=[], 
                         ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$\eta\:[\unit{\percent}]$"], 
                         ax_lims=[[], [-5,105]], 
                         ax_ticks=[[],np.arange(0,101,10)], 
                         fname=f"T1_eta_vs_V0_MPPT_{cont_type}")

        return eta, P_poc, P_mech
    
    def calc_task_2 (self, V_0, cont_type="a", plt_res = False):
        """Calculates the reactive power of the system-side circuit for which 
        the apparent power of the grid is purely active for specified wind 
        speed(-s) and MPPT control scheme
        
        Parameters:
            V_0 (scalar or array-like);
                Wind speed [m/s]
            cont_type (str):
                MPPT control scheme. Available control schemes are:
                - "a": Zero Power Factor: The phase current is in phase with  
                       the terminal voltage
                - "b": Zero Beta Angle: The phase current is in phase with 
                       the induced voltage  
        
        Returns:
            V_vscs_p (scalar or array-like);
                Voltage of the system-side VSC, referred to the high voltage 
                side [V]
            I_t1_p (scalar or array-like);
                Current of the system-side VSC, referred to the high voltage 
                side [A]
            S_vscs (scalar or array-like);
                Apparent power of the system-side VSC [VA]
            Q_vscs (scalar or array-like);
                Reactive power of the system-side VSC [VA]
        """
        
        P_mech = self.calc_P_mech (V_0)
        V_vscg, I_a, delta = self.calc_generator(V_0=V_0, cont_type=cont_type,
                                                 plt_res=plt_res)
        P_vscg = (3*V_vscg*I_a).real
        
        I_poc, P_poc_intsct  = self.solve_I_poc(P_vscg=P_vscg)
        V_vscs_p, I_t1_p, S_vscs, V_sec = self.calc_system_circuit_t2(I_poc=I_poc)
        P_poc = 3*33e3/np.sqrt(3)*I_poc
        
        if plt_res:
            #Plot System-side circuit
            V_poc = np.array([33e3/np.sqrt(3)]*len(V_vscs_p))
            self.plot_res(V_0, abs(np.vstack([V_vscs_p, V_poc, V_sec]))*4/33, 
                          plt_labels=[r"$V_\text{VSC-S}$", r"$V_\text{POC}$", 
                                      r"$V_\text{sec}$"], 
                         ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$V\:[\unit{\V}]$"], 
                         fname=f"T2_Sys_circuit_lowVolt_MPPT{cont_type}")
            self.plot_res(V_0, abs(np.vstack([V_vscs_p, V_poc, V_sec])), 
                          plt_labels=[r"$V_\text{VSC-S}$", r"$V_\text{POC}$", 
                                      r"$V_\text{sec}$"], 
                         ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$V\:[\unit{\V}]$"], 
                         fname=f"T2_Sys_circuit_highVolt_MPPT{cont_type}")
            
            #Plot Power
            self.plot_res(V_0, np.vstack((P_mech, P_vscg, 
                                          S_vscs.real, P_poc))/1e6, 
                          plt_labels=[r"$P_{mech}$", r"$P_\text{VSC-G}$",
                                      r"$P_\text{VSC-S}$", r"$P_{POC}$"], 
                          ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$P\:[\unit{\MW}]$"], 
                          ax_ticks=[[],np.arange(0,11,2)], 
                          fname=f"T2_P_vs_V0_MPPT{cont_type}")
            self.plot_res(V_0, np.vstack((abs(S_vscs), S_vscs.real, 
                                          S_vscs.imag))/1e6, 
                          plt_labels=[r"$S_\text{VSC-S}$", r"$P_\text{VSC-S}$",
                                      r"$Q_\text{VSC-S}$"], 
                          ax_labels=[r"$V_0\:[\unit{m/s}]$",
                                    r"$P\:[\unit{\MW}]\text{ or }"
                                    + r"S\:[\unit{\MW}]\text{ or }"
                                    + r"Q\:[\unit{\mega\V\A}]$"], 
                          ax_ticks=[[],np.arange(0,11,2)], 
                          fname=f"T2_S_vs_V0_MPPT{cont_type}")
        
        return V_vscs_p, I_t1_p, S_vscs, S_vscs.imag
        
    def plot_AB_t1 (self, P_vscg=10e6):
        """Plot the relationship between the system-side VSC voltage and the
        power grid voltage for a specified active power of the generator-side 
        VSC.
        
        Parameters:
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC [W]
        
        Returns:
            None
        """
        
        V_vscs = np.concatenate ((np.linspace(0.0001,2,50), 
                                 np.linspace(2,40000, 100)))
        
        V_poc = V_vscs*self.A_1 - P_vscg/3/V_vscs*self.B_1
        
        fig,ax = plt.subplots()
        ax.plot(V_vscs, abs(V_poc), ls = "-", 
                label=r"$V_{poc}$")
        ax.plot(V_vscs, V_poc.real, ls = ":", 
                label=r"$V_{poc}$")
        ax.plot(V_vscs, V_poc.imag, ls = ":", alpha=.7, 
                label=r"$V_{poc}$")
        
        ax.plot(V_vscs, self.A_1.real*V_vscs, ls = "--", c="k", 
                label=r"$\text{Re}\{\underline{V}_\text{VSC-S}\cdot\underline{A}\}$")
        ax.plot(V_vscs, self.A_1.imag*V_vscs, ls = "--", c="k", alpha=.7, 
                label=r"$\text{Im}\{\underline{V}_\text{VSC-S}\cdot\underline{A}\}$")
        
        ax.plot(V_vscs, self.B_1.real*P_vscg/3/V_vscs, ls = "-.", c="k", 
                label=r"$\text{Re}\{\frac{P_\text{VSC-G}}{3\cdot \underline{V}_\text{VSC-S}}\cdot\underline{B}\}$")
        ax.plot(V_vscs, self.B_1.imag*P_vscg/3/V_vscs, ls = "-.", c="k", alpha=.7, 
                label=r"$\text{Im}\{\frac{P_\text{VSC-G}}{3\cdot \underline{V}_\text{VSC-S}}\cdot\underline{B}\}$")
        
        ax.legend(loc = "upper left", bbox_to_anchor=(1.05,1))  
        ax.grid()
        ax.set_ylim([-20e3, 45e3])
        ax.set_xlabel(r"$V_\text{VSC-S}\:[\unit{V}]$")
        ax.set_ylabel(r"$V\:[\unit{V}]$")
        
        fname = self.exp_fld+"T1_V_poc_vs_V_vscs"
        fig.savefig(fname=fname+".svg")
        fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                       # inclusion in LaTeX
        plt.close(fig)
    
        
    def plot_AB_t2 (self):
        """Plot the relationship between the system-side VSC voltage and the
        power grid voltage for a specified active power of the generator-side 
        VSC.
        
        Parameters:
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC [W]
        
        Returns:
            None
        """
        
        I_poc_range = np.linspace(0,500, 1000)
        V_poc = 33e3/np.sqrt(3)
        
        S_vscs = 3*(V_poc*self.A_v + I_poc_range*self.B_v)\
                  *(V_poc*self.A_i.conjugate() 
                    + I_poc_range*self.B_i.conjugate())
        
        self.plot_res(I_poc_range, np.vstack((abs(S_vscs), S_vscs.real, 
                                      S_vscs.imag))/1e6, 
                      plt_labels=[r"$S_\text{VSC-S}$", r"$P_\text{VSC-S}$",
                                  r"$Q_\text{VSC-S}$"], 
                      ax_labels=[r"$I_\text{POC}\:[\unit{A}]$",
                                r"$P\:[\unit{\MW}]\text{ or }"
                                + r"S\:[\unit{\MW}]\text{ or }"
                                + r"Q\:[\unit{\mega\V\A}]$"], 
                      fname=f"T2_S_vs_I_poc")
    
    def plot_res(self, x, y, plt_labels=[], 
                 ax_labels=["",""], ax_lims=[(),()], 
                 ax_ticks=[(),()], fname=""):
        fig,ax = plt.subplots()
        
        if x.ndim>1 or y.ndim>1:
            if x.ndim == 1:
                x = np.tile(x.flatten(), (min(y.shape),1))
                y = y.reshape(min(y.shape), -1)
            if y.ndim == 1:
                y = np.tile(y.flatten(), (min(x.shape),1))
                x = x.reshape(min(x.shape), -1)
            
            if len(plt_labels)==0:
                plt_labels = ["var"+str(i) for i in range(x.shape[0])]
            elif not len(plt_labels)==x.shape[0]:
                raise ValueError("Invalid number of plt_labels for" 
                                 + " number of input dimensions")
            
            for i in range(x.shape[0]):
                ax.plot(x[i,:], y[i,:], label = plt_labels[i])
        else:   
            ax.plot(x, y)
        
        if len(ax_labels[0])>0:
            ax.set_xlabel(ax_labels[0])
        if len(ax_labels[1])>0:
            ax.set_ylabel(ax_labels[1])
        
        if len(ax_lims[0])>0:
            ax.set_xlim(ax_lims[0])
        if len(ax_lims[1])>0:
            ax.set_ylim(ax_lims[1])
            
        if len(ax_ticks[0])>0:
            ax.set_xticks(ax_ticks[0])
        if len(ax_ticks[1])>0:
            ax.set_yticks(ax_ticks[1])
        ax.grid(zorder=1)
        
        if x.ndim>1:
            ax.legend()
        
        if fname:
            fname = self.exp_fld+fname
        elif len(plt_labels)>0:
            if np.isscalar(plt_labels):
                fname = self.exp_fld + plt_labels
            else:
                fname = self.exp_fld + plt_labels[0]
        fig.savefig(fname=fname+".svg")
        fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                       # inclusion in LaTeX
        plt.close(fig)
        
#%% Main       
if __name__ == "__main__":
    WFC = WindFarmCalculator()  
    V_0 = np.arange(2,10.5,.5)
    
    #%% Task 1
    WFC.plot_AB_t1()
    eta_a, P_poc_a, P_mech_a = WFC.calc_efficiency(V_0=V_0, cont_type="a", 
                                                   plt_res=True)
    eta_b, P_poc_b, P_mech_b = WFC.calc_efficiency(V_0=V_0, cont_type="b", 
                                                   plt_res=True)
    
    WFC.plot_res(V_0, np.vstack((eta_a, eta_b))*100, 
                 plt_labels=["Zero Power angle", "Zero beta angle"], 
                 ax_labels=[r"$V_0\:[\unit{m/s}]$",
                            r"$\eta\:[\unit{\percent}]$"], 
                 ax_lims=[[], [-5,105]], 
                 ax_ticks=[[],np.arange(0,101,10)], 
                 fname=f"T1_eta_vs_V0")
    
    #%% Task 2
    WFC.plot_AB_t2()
    V_vscs_p_a, I_t1_p_a, S_vscs_a, Q_vscs_a = WFC.calc_task_2 (V_0=V_0, 
                                                                cont_type="a", 
                                                                plt_res = True)
    V_vscs_p_b, I_t1_p_b, S_vscs_b, Q_vscs_b = WFC.calc_task_2 (V_0=V_0, 
                                                                cont_type="b", 
                                                                plt_res = True)
    
    WFC.plot_res(V_0, np.vstack((Q_vscs_a, Q_vscs_b)), 
                 plt_labels=["Zero Power angle", "Zero beta angle"], 
                 ax_labels=[r"$V_0\:[\unit{m/s}]$",
                            r"$Q\:[\unit{\mega\V\A}]$"], 
                 fname=f"T2_Q_vs_V0")
