import tensorflow as tf
import time

# Check if TensorFlow detects a GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPU detected:", gpus)
else:
    print("No GPU detected. Ensure CUDA and cuDNN are properly installed.")
    exit()
gpu_device = gpus[0]
tf.config.experimental.set_memory_growth(gpu_device, True)

# Define a stress test function
def gpu_stress_test(num_iterations=1000, matrix_size=2048):
    print(f"Running stress test: {num_iterations} iterations with {matrix_size}x{matrix_size} matrices.")
    
    # Create random matrices for multiplication
    matrix_a = tf.random.uniform((matrix_size, matrix_size))
    matrix_b = tf.random.uniform((matrix_size, matrix_size))
    
    start_time = time.time()
    
    # Perform matrix multiplications
    for i in range(num_iterations):
        tf.linalg.matmul(matrix_a, matrix_b)
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{num_iterations} completed.")
    
    end_time = time.time()
    print(f"Stress test completed in {end_time - start_time:.2f} seconds.")

# Run the GPU stress test
gpu_stress_test(num_iterations=10000, matrix_size=4096)
