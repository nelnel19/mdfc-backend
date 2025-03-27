import subprocess

# Start the first Uvicorn server
server1 = subprocess.Popen(["uvicorn", "analyze:app", "--port", "8001"])

# Start the second Uvicorn server
server2 = subprocess.Popen(["uvicorn", "main:app", "--port", "8000"])

# Wait for both servers to finish
server1.communicate()
server2.communicate()

