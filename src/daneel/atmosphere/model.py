import os
import subprocess
import yaml


class AtmosphericForwardModel:

    import os


    def __init__(self, parameters):


        self.input_file = os.path.abspath(parameters)
        self.base_dir = os.path.dirname(self.input_file)

        self.output_dir = os.path.join(self.base_dir, "outputs/")
        self.output = os.path.join(self.output_dir, "transmission.h5")

    def build(self):
        """Prepare the forward model (directories, checks)."""

        if not os.path.isfile("parameters.par"):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def trasnformation(self):
        """Convert YAML input into TauREx .par format."""

        with open(self.input_file) as f:
            params = yaml.safe_load(f)

        par_file = "parameters.par"
        with open(par_file, "w") as f:
            for section, values in params.items():
                f.write(f"[{section}]\n")

                if section == "Chemistry":
                # Write top-level Chemistry keys
                    for k in ["chemistry_type", "fill_gases", "ratio"]:
                        val = values.get(k)
                        if val is not None:
                            if k == "fill_gases" and isinstance(val, list):
                                f.write(f"{k} = {','.join(val)}\n")
                            else:
                                f.write(f"{k} = {val}\n")
                    f.write("\n")

                # Write species as [[species_name]] sections
                    species = values.get("species", {})
                    for gas_name, gas_info in species.items():
                        f.write(f"[[{gas_name}]]\n")
                        for sub_k, sub_v in gas_info.items():
                            f.write(f"{sub_k} = {sub_v}\n")
                        f.write("\n")
                    continue
                        # Special handling for Model
                if section == "Model":
                    for k, v in values.items():
                        if isinstance(v, dict):
                            f.write(f"[[{k}]]\n")
                            for sub_k, sub_v in v.items():
                                if isinstance(sub_v, list):
                                    f.write(f"{sub_k} = {','.join(map(str, sub_v))}\n")
                                else:
                                    f.write(f"{sub_k} = {sub_v}\n")
                            f.write("\n")
                        elif isinstance(v, list) and not v:
                            f.write(f"[[{k}]]\n\n")
                        else:
                            f.write(f"{k} = {v}\n")
                    f.write("\n")
                    continue

                if section == "Fitting":
                    for param_name, param_info in values.items():
                        fit_val = param_info.get("fit", False)
                        bounds_val = param_info.get("bounds", [])
                        f.write(f"{param_name}:fit = {str(fit_val)}\n")
                        if bounds_val:
                            f.write(f"{param_name}:bounds = {','.join(map(str, bounds_val))}\n")
                        f.write("\n")
                    continue

            # Generic handling for other sections
                if isinstance(values, dict):
                    for k, v in values.items():
                        if isinstance(v, dict):
                            v_flat = [str(sub_v) for sub_v in v.values()]
                            f.write(f"{k} = {','.join(v_flat)}\n")
                        elif isinstance(v, list):
                            f.write(f"{k} = {','.join(map(str, v))}\n")
                        else:
                            f.write(f"{k} = {v}\n")
                elif isinstance(values, list):
                    for item in values:
                        f.write(f"{item}\n")
                else:
                    f.write(f"{values}\n")

                f.write("\n") 

        print(f"TauREx parameter file saved as {par_file}")

    def run(self):
        """Run the atmospheric forward model."""
        print(  "### START RUN ###")
        try:

            
            result = subprocess.run([
                "taurex",
                "-i", "parameters.par",
                "-o", self.output
            ])

    
            print("TauREx completed successfully.")
            print(f"Output saved to: {self.output}")

        except subprocess.CalledProcessError as e:
            print("TauREx failed:")
            print(e.stderr)
            raise

        except FileNotFoundError:
            print("TauREx executable not found in PATH.")
            raise

    def get_output_path(self):
        return self.output



def main(path):
    
        print("\n",path)

        model = AtmosphericForwardModel(path)
        model.trasnformation()
        model.build()
        model.run()

        print("MWMWMMWMWMWMWMMWMWMWM file saved MWMWMMWMWMWMWMMWMWMWM")