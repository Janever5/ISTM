import requests
import os
import shutil
import pandas as pd
import numpy as np

BASE_URL = "http://localhost:5000"
TEST_DIR = "qtbfs_test_data"

def create_dummy_csv(filepath, duration_s=10, sampling_rate=1000, frequency=2, amplitude=50):
    """Creates a dummy CSV with a sine wave."""
    num_points = int(duration_s * sampling_rate)
    time = np.linspace(0, duration_s, num_points, endpoint=False)
    # Create a sine wave with some noise
    resistance = 1000 + amplitude * np.sin(2 * np.pi * frequency * time) + np.random.randn(num_points) * 5
    df = pd.DataFrame({
        'index': range(num_points),
        'time': time,
        'resistance': resistance
    })
    df.to_csv(filepath, index=False)

def setup_test_data():
    """Creates all necessary dummy files for testing."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    
    state0_dir = os.path.join(TEST_DIR, 'state0')
    current_dir = os.path.join(TEST_DIR, 'current')
    os.makedirs(state0_dir)
    os.makedirs(current_dir)

    # Define files to create for both states as per 传感.md
    angle_files = [f"angle_{deg}.csv" for deg in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 120]]
    speed_files = [f"speed_{deg}deg_{s}s.csv" for deg in [30, 60, 90] for s in [1, 2, 3]]
    
    all_files = angle_files + speed_files

    # Create files for state0 (healthy reference)
    print("Creating state0 data...")
    for filename in all_files:
        create_dummy_csv(os.path.join(state0_dir, filename), amplitude=100) # Higher amplitude for healthy
    
    # Create files for current_state (impaired)
    print("Creating current_state data...")
    for filename in all_files:
        create_dummy_csv(os.path.join(current_dir, filename), amplitude=60) # Lower amplitude for impaired
        
    print(f"Created {len(all_files)*2} dummy files in {TEST_DIR}")

def run_test():
    """Uploads files and calls the QTBFS API."""
    state0_dir = os.path.join(TEST_DIR, 'state0')
    current_dir = os.path.join(TEST_DIR, 'current')

    state0_files = [('state0_files', (f, open(os.path.join(state0_dir, f), 'rb'), 'text/csv')) for f in os.listdir(state0_dir)]
    current_files = [('current_files', (f, open(os.path.join(current_dir, f), 'rb'), 'text/csv')) for f in os.listdir(current_dir)]

    all_files_to_upload = state0_files + current_files

    try:
        print("\nSending request to /api/qtbfs_calculate...")
        response = requests.post(f"{BASE_URL}/api/qtbfs_calculate", files=all_files_to_upload)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ API call successful!")
                score_result = result.get('result', {})
                print(f"  Total Score: {score_result.get('total_score')}")
                print(f"  Stage: {score_result.get('stage')}")
                print(f"  Domain I: {score_result.get('domain_I', {}).get('total')}")
                print(f"  Domain II: {score_result.get('domain_II', {}).get('total')}")
                print(f"  Domain III: {score_result.get('domain_III', {}).get('total')}")
            else:
                print(f"❌ API call failed: {result.get('error')}")
                print(f"   Details: {result.get('details')}")
        else:
            print("❌ Request failed. Server response:")
            print(response.text)

    except Exception as e:
        print(f"An error occurred during the request: {e}")
    finally:
        # Close all opened files
        for _, (f_name, f_obj, _) in all_files_to_upload:
            f_obj.close()

def cleanup():
    """Removes the test data directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    print(f"\nCleaned up test directory: {TEST_DIR}")

if __name__ == "__main__":
    setup_test_data()
    run_test()
    cleanup()