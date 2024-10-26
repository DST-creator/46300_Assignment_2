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
        self.Z_t3_p = a**2 * complex(1214e-3, self.omega*55e-3)
        self.Z_c1 = complex(8, self.omega*26e-3)
        self.Z_c2 = complex(0, -1/(self.omega*2e-6))
        
        self.calc_E_a = lambda omega_G: omega_G*self.Psi
        self.calc_omega_G = lambda V_0: self.omega_G_rtd*V_0/self.V_rtd
        
        self.A = (1-(self.Z_t2+self.Z_c1)/self.Z_t3_p)
        self.B = self.Z_t1_p-(self.Z_t2+self.Z_c1)*(self.Z_t1_p/self.Z_t3_p+1)
        
        #Misc
        self.exp_fld = exp_fld
        if not os.path.isdir(self.exp_fld):
            os.mkdir(self.exp_fld)

#%% Task 1
    
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

    def calc_generator(self, V_0, cont_type="a"):
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
        
        R_s = 25e-3
        X_L = omega_G*10e-6
        
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
        
        return V_vscg, I_a, delta
    
    def calc_system_circuit(self, P_vscg, V_vscs):
        """Calculate the grid voltage and current based on the active power of 
        the generator-side voltage source converter (VSC) and an assumed value 
        of the voltage of the system side VSC.
        
        Parameters:
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC [W]
            V_vscs (scalar or array-like);
                Voltage of the system-side VSC [V]
            
            
        Returns:
            V_poc (scalar or array-like);
                The power grid voltage [V]
            I_poc  (scalar or array-like);
                The power grid current [A]
        """
        
        if not type (V_vscs) == complex:
            V_vscs = complex(V_vscs, 0)
        
        I_t1 = P_vscg/(3*V_vscs)
        I_t3 = (V_vscs-I_t1*self.Z_t1_p)/self.Z_t3_p
        I_t2 = I_t3 - I_t1
        V_poc = I_t3*self.Z_t3_p-I_t2*self.Z_t2-I_t2*self.Z_c1
        I_c2 = V_poc/self.Z_c2
        I_poc = I_t2-I_c2
        
        return V_poc, I_poc
    
    def plot_AB (self, P_vscg=10e6):
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
        
        V_poc = V_vscs*self.A - P_vscg/3/V_vscs*self.B
        
        fig,ax = plt.subplots()
        ax.plot(V_vscs, abs(V_poc), ls = "-", 
                label=r"$V_{poc}$")
        
        ax.plot(V_vscs, self.A.real*V_vscs, ls = "--", c="k", 
                label=r"$\text{Re}\{\underline{V}_\text{VSC-S}\cdot\underline{A}\}$")
        ax.plot(V_vscs, self.A.imag*V_vscs, ls = "--", c="k", alpha=.7, 
                label=r"$\text{Im}\{\underline{V}_\text{VSC-S}\cdot\underline{A}\}$")
        
        ax.plot(V_vscs, self.B.real*P_vscg/3/V_vscs, ls = "-.", c="k", 
                label=r"$\text{Re}\{\frac{P_\text{VSC-G}}{3\cdot \underline{V}_\text{VSC-S}}\cdot\underline{B}\}$")
        ax.plot(V_vscs, self.B.imag*P_vscg/3/V_vscs, ls = "-.", c="k", alpha=.7, 
                label=r"$\text{Im}\{\frac{P_\text{VSC-G}}{3\cdot \underline{V}_\text{VSC-S}}\cdot\underline{B}\}$")
        
        ax.legend(loc = "upper left", bbox_to_anchor=(1.05,1))  
        ax.grid()
        ax.set_ylim([-20e3, 45e3])
        ax.set_xlabel(r"$V_\text{VSC-S}\:[\unit{V}]$")
        ax.set_ylabel(r"$V\:[\unit{V}]$")
        
        fname = self.exp_fld+"V_poc_vs_V_vscs"
        fig.savefig(fname=fname+".svg")
        fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                       # inclusion in LaTeX
        plt.close(fig)
    
    def solve_V_vscs (self, P_vscg):
        """Find the systems-side VSC voltage for which the power grid voltage 
        becomes equal to its nominal value of 33kV  for a specified active 
        power of the generator-side VSC.
        
        Parameters:
            P_vscg (scalar or array-like);
                Active power of the generator-side VSC [W]
        
        Returns:
            V_vscs (scalar or array-like);
                Voltage of the system-side VSC [V]
        """
        
        V_vscs_range = np.linspace(1,50e3, 200)
        
        V_poc = abs(V_vscs_range*self.A 
                    - P_vscg.reshape(-1,1)/3/V_vscs_range*self.B)

        V_vscs = np.zeros(len(P_vscg))
        for i in range(len(P_vscg)):
            V_vscs = intersection(V_vscs_range, V_poc[i,:],
                                  [0,50e3], [33e3, 33e3])[0][0]
            
        return V_vscs  

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
        V_vscg, I_a, delta = self.calc_generator(V_0=V_0, cont_type=cont_type)
        P_vscg = (3*V_vscg*I_a).real
        
        V_vscs = self.solve_V_vscs(P_vscg=P_vscg)
        V_poc, I_poc = self.calc_system_circuit(P_vscg=P_vscg, V_vscs=V_vscs)
        
        P_poc = (3*V_vscg*I_a).real
        
        eta = P_poc/P_mech
        
        if plt_res:
            fig,ax1 = plt.subplots()
            
            line1 = ax1.plot(V_0, eta*100, label=r"$\eta$")
            
            ax2 = ax1.twinx()
            line2 =ax2.plot(V_0, P_mech/1e6, ls="--", label=r"$P_{mech}$")
            line3 =ax2.plot(V_0, P_mech/1e6, ls="-.", label=r"$P_{POC}$")
            
            #Add the three lines to the legend
            lns = line1 + line2 + line3
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc="best")
             
            ax1.grid()
            ax1.set_xlabel(r"$V_0\:[\unit{m/s}]$")
            ax1.set_ylabel(r"$\eta\:[\unit{\percent}]$")
            ax2.set_ylabel(r"$P\:[\unit{\MW}]$")
            
            # ax1.set_ylim([-5,105])
            # ax1.set_yticks(np.arange(0,101,10))
            
            fname = self.exp_fld+f"eta_MPPT{cont_type}"
            fig.savefig(fname=fname+".svg")
            fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
            fig.savefig(fname+".pgf")                     # Save PGF file for text 
                                                           # inclusion in LaTeX
            plt.close(fig)

        return eta, P_poc, P_mech
        
    
if __name__ == "__main__":
    WFC = WindFarmCalculator()    
    eta, P_poc, P_mech = WFC.calc_efficiency(np.arange(1,30), "a", plt_res=True)

