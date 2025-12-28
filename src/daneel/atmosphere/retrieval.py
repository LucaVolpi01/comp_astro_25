import h5py
import numpy as np
import subprocess
import os

class AtmosphericRetrieve:
  
    def __init__(self, path):
        self.input_file = os.path.abspath(path)
        self.base_dir = os.path.dirname(self.input_file)
        self.output_dir = os.path.join(self.base_dir, "outputs/")
        self.transmission_path = os.path.join(self.output_dir, "transmission.h5")
        self.output_file = os.path.join(self.output_dir, "transmission_with_errors.h5")
        
        # CREATE DIRECTORY 
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"./retrieval_plots/", exist_ok=True)
    
    def create_file_with_error(self):
        """Generate observed spectrum with noise from transmission model
            - file created : observed_spectrum.dat1          """
        
        FIXED_ERROR = 1e-5
        
        if not os.path.exists(self.transmission_path):
            print(f"Error: Transmission file not found: {self.transmission_path}")
            return False
        
        
        with h5py.File(self.transmission_path, 'r') as f:
                wl = f['Output/Spectra/native_wlgrid'][:]
                spectrum = f['Output/Spectra/native_spectrum'][:]
            
            # Fixed absolute uncertainty
        noise = np.full_like(spectrum, FIXED_ERROR)
        observed_spectrum = spectrum + np.random.normal(0, noise)
            
            # Save as text file for TauREx retrieval
        data = np.column_stack([wl, observed_spectrum, noise])
        output_txt = os.path.join(self.base_dir, f"observed_spectrum.dat")
        np.savetxt(
                output_txt,
                data,
                fmt='%.8e',
                header='wavelength(um) transit_depth error')
            
        print(f"✓ Created observed spectrum: {output_txt}")
        return True
    
    
    def retrieval(self):
        
        """Run atmospheric retrieval 
                -input: parameters.par
                -retrieval output:retrieval_output.h5
                -output: posterior plot in folder "retrieval_plots/" 
                """
        print("Running retrieval...")
        
        plot_dir = os.path.join(self.base_dir, "retrieval_plots")
        
        
            
            # Run TauREx retrieval
        subprocess.run([
                "taurex", 
                "-i", "parameters.par",   
                "-o", "outputs",
                "-S", "retrieval_output.h5",
                "--retrieval"
            ], check=True)
            
           
        print(f" Retrieval complete")
        """
            # Generate plots
        print("Generating plots...")
        subprocess.run([    
                "taurex-plot", 
                "-i", "retrieval_output.h5",
                "-o", plot_dir,
                "-P"
            ], check=True)
            
        print(f" Plots saved to: {plot_dir}")
        return True
        """
   

def main(path):
    """Main function to run atmospheric retrieval workflow"""
    
    
    # Create model instance
    model = AtmosphericRetrieve(path)
    
    # Step 1: Create observed spectrum with errors
    print("\nStep 1: Creating observed spectrum with errors...")
  #  model.create_file_with_error()
 
    
    # Step 2: Run retrieval
    print("\nStep 2: Running atmospheric retrieval...")
    model.retrieval()
    
    print("✓ Atmospheric retrieval complete!")
    print(f"Results saved in: {model.output_dir}")

if __name__ == "__main__":

    main(parameter_file)