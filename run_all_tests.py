import subprocess
import sys
import time

def run_test(script_name):
    print(f"\n{'='*80}")
    print(f"Running {script_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            print(f"\n{script_name} completed successfully!")
        else:
            print(f"\n{script_name} failed with return code {result.returncode}")
    except Exception as e:
        print(f"\nError running {script_name}: {str(e)}")
    
    # Wait a bit between tests to allow GPU memory to clear
    time.sleep(5)

# List of test scripts to run
test_scripts = [
    "yolov5_test.py",
    "yolov8_test.py",
    "yolov11_test.py",
    "faster-R-CNN_test.py",
    "Cascade R-CNN_test.py"
]

# Run each test script
for script in test_scripts:
    run_test(script)

print("\nAll tests completed!") 