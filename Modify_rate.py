import xml.etree.ElementTree as ET

class XMLRateModifier:
    def __init__(self, file_path, reduction_percent):
        self.file_name = file_name
        self.reduction_percent = reduction_percent
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()

    def modify_rates(self):
        #Reduce 'rate' values by the set percentage
        factor = 1 - (self.reduction_percent / 100)

        for stim in self.root.findall(".//InjectionStim[@specieID='Ca']"): #**only** for specieID='Ca'; can be chnaged to make it general
            rate_element = stim.find("rate")
            if rate_element is not None:
                original_rate = float(rate_element.text)
                new_rate = original_rate * factor  # Apply reduction
                rate_element.text = f"{new_rate:.4f}" #write new rate with only 4 decimals

    def save_modified_file(self, output_file):
        #Save the modified XML to a new file.
        self.tree.write(output_file)
        print(f"Modified XML saved as '{output_file}'")

# Usage
file_name = "Stim_Cof-4trains_spaced80.xml"  # Replace with your actual XML file, CAN MAKE IT RUN MULTIPLE AT ONCE
output_file = file_name.split('.')[0]+"_PKA.xml" #general name as well
reduction_percent = 20  # Change this value as needed

modifier = XMLRateModifier(file_name, reduction_percent)
modifier.modify_rates()
modifier.save_modified_file(output_file)