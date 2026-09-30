import os
import sys
from translator.data_collector import DataCollector

def main():
    """
    Collect data for sign language recognition
    """
    print("Sign Language Data Collection")
    print("-----------------------------")
    
    # Initialize data collector
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    collector = DataCollector(data_dir=data_dir)
    
    # List available signs
    print("Available signs:")
    for i, sign in enumerate(collector.signs):
        print(f"{i+1}. {sign}")
    
    # Get sign to collect data for
    while True:
        try:
            sign_index = int(input("\nEnter the number of the sign to collect data for (0 to exit): ")) - 1
            
            if sign_index == -1:
                print("Exiting data collection.")
                return
            
            if 0 <= sign_index < len(collector.signs):
                sign = collector.signs[sign_index]
                break
            else:
                print(f"Please enter a number between 1 and {len(collector.signs)}.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of samples
    while True:
        try:
            num_samples = int(input(f"Enter the number of samples to collect for '{sign}' (recommended: 100): "))
            if num_samples > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Collect data
    print(f"\nCollecting {num_samples} samples for sign '{sign}'...")
    print("Position your hand in the webcam view and make the sign.")
    print("Press any key to start collection, or 'q' to quit.")
    
    # Start collection
    collector.collect_data(sign, num_samples)
    
    print("\nData collection complete.")
    print(f"Data saved to {os.path.join(data_dir, sign)}")

if __name__ == "__main__":
    main()

