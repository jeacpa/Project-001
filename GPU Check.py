import GPUtil

gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(
        f"GPU ID: {gpu.id}, Name: {gpu.name}, Load: {gpu.load * 100:.2f}%, Memory Usage: {gpu.memoryUsed / gpu.memoryTotal * 100:.2f}%")